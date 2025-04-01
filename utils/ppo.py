from __future__ import annotations

from collections.abc import Hashable
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from frozendict import frozendict

from utils.get_action_and_value import get_action_and_value2
from utils.utils import ActorTrainState

if TYPE_CHECKING:
    import chex
    import numpy as np
    from numpy.typing import NDArray

    from config_types.args_types import CLIArgs
    from mlp import Actor_MLP, Actor_MLP_Continuous, Critic_MLP
    from ray_utilities.jax.jax_model import PureJaxModelProtocol
    from sdt import Actor_SDT, Critic_SDT
    from sympol import SYMPOL_RL
    from utils.utils import Storage, TrainState

    _Actor = PureJaxModelProtocol | Actor_MLP | Actor_MLP_Continuous | Actor_SDT | SYMPOL_RL
    _Critic = Critic_MLP | Critic_SDT

    # keep signature complete
    jax.jit = lambda func, *args, **kwargs: func  # noqa: ARG005


def _compute_gae_once(
    carry: chex.Array,
    inp: tuple[jax.Array, chex.Array, chex.Array, jax.Array | float],
    gamma: chex.Numeric,
    gae_lambda: chex.Numeric,
) -> tuple[chex.Array, chex.Array]:
    advantages = carry
    nextdone, nextvalues, curvalues, reward = inp
    nextnonterminal = 1.0 - nextdone

    delta = reward + gamma * nextvalues * nextnonterminal - curvalues
    advantages = delta + gamma * gae_lambda * (nextnonterminal * advantages)
    return advantages, advantages


@partial(jax.jit, static_argnames=("critic", "args"))
def compute_gae(
    critic_state: TrainState,
    next_obs: np.ndarray,
    next_done: NDArray[np.bool_] | jnp.ndarray,
    storage: Storage,
    *,
    critic: _Critic,
    args: CLIArgs,
):
    # Alternatively make compute_gae_once a static arg
    compute_gae_once = partial(_compute_gae_once, gamma=args.gamma, gae_lambda=args.gae_lambda)
    next_value = critic.apply(critic_state.params, next_obs).squeeze()  # pyright: ignore[reportAttributeAccessIssue]
    if args.n_envs <= 1:  # add batch dimension
        next_value = next_value[jnp.newaxis]
    advantages = jnp.zeros((args.n_envs,))
    # turn back to
    dones = jnp.concatenate([storage.dones, next_done[jnp.newaxis]], axis=0)
    values = jnp.concatenate([storage.values, next_value[jnp.newaxis]], axis=0)
    _, advantages = jax.lax.scan(
        compute_gae_once, advantages, (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True
    )
    storage = storage.replace(  # type: ignore[attr-defined]
        advantages=advantages,
        returns=advantages + storage.values,
    )
    return storage


@partial(jax.jit, static_argnames=("action_type", "actor", "critic", "actor_state_indices", "args"))
def ppo_loss_base(
    actor_state_params,
    critic_state_params,
    x,
    a,
    logp,
    mb_advantages,
    mb_returns,
    *,
    action_type: Literal["discrete", "continuous"],
    actor: _Actor,
    critic: _Critic,
    actor_state_indices,
    args: CLIArgs,
):
    """
    Attention:
        Make sure that args is hashable and has __eq__ to make it correctly static
    """
    if not isinstance(args, Hashable):
        raise TypeError("args must be hashable")
    # NEW
    newlogprob, entropy, newvalue = get_action_and_value2(
        actor_state_params,
        critic_state_params,
        x,
        a,
        action_type=action_type,
        actor=actor,
        critic=critic,
        actor_state_indices=actor_state_indices,
    )
    logratio = newlogprob - logp
    ratio = jnp.exp(logratio)
    approx_kl = ((ratio - 1) - logratio).mean()

    if args.norm_adv:
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

    # Policy loss
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # Value loss
    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))


ppo_loss_base_grad_fn = jax.value_and_grad(ppo_loss_base, argnums=(0, 1), has_aux=True)


@partial(
    jax.jit, static_argnames=("minibatch_size", "n_update_epochs", "args", "actor", "critic", "actor_state_indices")
)
def update_ppo(
    actor_state: ActorTrainState,
    critic_state: TrainState,
    storage: Storage,
    key: chex.PRNGKey,
    accumulate_gradients_every: int,
    *,
    minibatch_size: int,
    n_update_epochs: int,
    args: CLIArgs,
    actor: _Actor,
    critic: _Critic,
    actor_state_indices,
) -> tuple[ActorTrainState, TrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, chex.PRNGKey]:
    def update_epoch(carry: tuple[ActorTrainState, TrainState, chex.PRNGKey], _unused_key):
        actor_state, critic_state, key = carry
        key, subkey = jax.random.split(key)

        def flatten(x):
            return x.reshape((-1,) + x.shape[2:])

        # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
        def convert_data(x: jnp.ndarray):
            num_minibatches = int(np.floor(x.shape[0] / minibatch_size))
            size = num_minibatches * minibatch_size
            x = jax.random.permutation(subkey, x)[:size]
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        flatten_storage = jax.tree_map(flatten, storage)
        shuffled_storage = jax.tree_map(convert_data, flatten_storage)

        def update_minibatch(
            carry: tuple[ActorTrainState, TrainState], minibatch: Storage
        ) -> tuple[
            tuple[ActorTrainState, TrainState],
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Any],
        ]:
            actor_state, critic_state = carry
            (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), (actor_grads, critic_grads) = ppo_loss_base_grad_fn(
                actor_state.params,
                critic_state.params,
                minibatch.obs,
                minibatch.actions,
                minibatch.logprobs,
                minibatch.advantages,
                minibatch.returns,
                action_type=args.action_type,
                actor=actor,
                critic=critic,
                actor_state_indices=actor_state_indices,
                args=args,
            )
            critic_state: TrainState = critic_state.apply_gradients(grads=critic_grads)
            actor_grad_accum = jax.tree_util.tree_map(lambda x, y: x + y, actor_grads, actor_state.grad_accum)
            actor_state: ActorTrainState = actor_state.apply_gradients(grads=actor_grads)

            def update_fn():
                grads = jax.tree_util.tree_map(lambda x: x / accumulate_gradients_every, actor_grad_accum)
                new_state = actor_state.apply_gradients(
                    grads=grads,
                    grad_accum=jax.tree_util.tree_map(jnp.zeros_like, grads),
                )
                return new_state

            actor_state = jax.lax.cond(
                actor_state.step % accumulate_gradients_every == 0,
                lambda _: update_fn(),
                lambda _: actor_state.replace(grad_accum=actor_grad_accum, step=actor_state.step + 1),
                None,
            )

            return (actor_state, critic_state), (
                loss,
                pg_loss,
                v_loss,
                entropy_loss,
                approx_kl,
                actor_grad_accum,
            )

        (actor_state, critic_state), (loss, pg_loss, v_loss, entropy_loss, approx_kl, actor_grad_accum) = jax.lax.scan(
            update_minibatch, (actor_state, critic_state), shuffled_storage
        )
        return (actor_state, critic_state, key), (
            loss,
            pg_loss,
            v_loss,
            entropy_loss,
            approx_kl,
            actor_grad_accum,
        )

    (actor_state, critic_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, _actor_grad_accum) = (
        jax.lax.scan(update_epoch, (actor_state, critic_state, key), (), length=n_update_epochs)
    )
    return actor_state, critic_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key
