from __future__ import annotations
from functools import partial

import distrax
import jax
import jax.numpy as jnp
from typing import TYPE_CHECKING, Any, Literal

from ray_utilities.jax.jax_model import PureJaxModelProtocol


if TYPE_CHECKING:
    from flax.core import FrozenDict as FlaxFrozenDict
    from flax.typing import FrozenVariableDict
    from sympol import SYMPOL_RL
    from sdt import Actor_SDT, Critic_SDT
    from mlp import Actor_MLP, Critic_MLP, Actor_MLP_Continuous
    import numpy as np
    from utils.utils import ActorTrainState, Storage, TrainState
    import chex

    _Actor = PureJaxModelProtocol | Actor_MLP | Actor_MLP_Continuous | Actor_SDT | SYMPOL_RL
    _Critic = Critic_MLP | Critic_SDT

    # Overwritting at typing level keeps signature complete
    jax.jit = lambda func, *args, **kwargs: func


@partial(jax.jit, static_argnames=("action_type", "actor", "critic", "actor_state_indices"))
def get_action_and_value(
    actor_state_params: FrozenVariableDict,
    critic_state: TrainState,
    next_obs: np.ndarray,
    next_done: np.ndarray,
    storage: Storage,
    step: int,
    key: chex.PRNGKey,
    *,
    action_type: Literal["discrete", "continuous"],
    actor: _Actor,
    critic: _Critic,
    actor_state_indices: dict,
) -> tuple[Storage, Any | chex.Array, chex.PRNGKey]:
    """sample action, calculate value, logprob, entropy, and update storage"""
    if action_type == "discrete":
        action_logits: jax.Array = actor.apply(actor_state_params, next_obs, indices=actor_state_indices)  # pyright: ignore[reportAssignmentType]
        action_distribution = distrax.Categorical(logits=action_logits)
        value: jax.Array = critic.apply(critic_state.params, next_obs)  # pyright: ignore[reportAssignmentType]

        # Sample discrete actions from Normal distribution
        key, subkey = jax.random.split(key)
        action = action_distribution.sample(seed=subkey)

        logprob = action_distribution.log_prob(action)  # .sum(-1)
        storage = storage.replace(  # type: ignore[attr-defined]
            obs=storage.obs.at[step].set(next_obs),
            dones=storage.dones.at[step].set(next_done),
            actions=storage.actions.at[step].set(action),
            logprobs=storage.logprobs.at[step].set(logprob),
            values=storage.values.at[step].set(value.squeeze()),
        )
    else:
        # result: layer_output, log_std
        result: tuple[jax.Array, chex.Array] = actor.apply(actor_state_params, next_obs, indices=actor_state_indices)  # pyright: ignore[reportAssignmentType]
        action_distribution = distrax.MultivariateNormalDiag(result[0], jnp.exp(result[1]))  # pyright: ignore[reportArgumentType]

        value: jax.Array = critic.apply(critic_state.params, next_obs)  # pyright: ignore[reportAssignmentType]

        # Sample continuous actions from Normal distribution
        key, subkey = jax.random.split(key)
        action = action_distribution.sample(seed=subkey)
        logprob = action_distribution.log_prob(action)  # .sum(-1)

        storage = storage.replace(  # type: ignore[attr-defined]
            obs=storage.obs.at[step].set(next_obs),
            dones=storage.dones.at[step].set(next_done),
            actions=storage.actions.at[step].set(action),
            logprobs=storage.logprobs.at[step].set(logprob),
            values=storage.values.at[step].set(value.squeeze()),
        )

    return storage, action, key


# or use partial to bind action_type, actor, critic, ... like compute_gae_once


@partial(jax.jit, static_argnames=("action_type", "actor", "critic", "actor_state_indices"))
def get_action_and_value2(
    actor_state_params: FlaxFrozenDict,
    critic_state_params: FlaxFrozenDict,
    x: np.ndarray,
    action: np.ndarray,
    *,
    action_type: Literal["discrete", "continuous"],
    actor: _Actor,
    critic: _Critic,
    actor_state_indices: dict,
) -> tuple[chex.Array, chex.Numeric, jax.Array]:
    """calculate value, logprob of supplied `action`, and entropy"""
    if action_type == "discrete":
        if TYPE_CHECKING:
            assert isinstance(actor, Actor_MLP)
        logits: jax.Array = actor.apply(actor_state_params, x, indices=actor_state_indices)  # pyright: ignore[reportAssignmentType]
        value = critic.apply(critic_state_params, x).squeeze()  # pyright: ignore[reportAttributeAccessIssue]

        action_distribution = distrax.Categorical(logits=logits)
        logprob = action_distribution.log_prob(action)
        entropy = action_distribution.entropy()
    else:
        # result: layer_output, log_std
        result: tuple[jax.Array, chex.Array] = actor.apply(actor_state_params, x, indices=actor_state_indices)  # pyright: ignore[reportAssignmentType]
        action_distribution = distrax.MultivariateNormalDiag(result[0], jnp.exp(result[1]))  # pyright: ignore[reportArgumentType]

        value = critic.apply(critic_state_params, x).squeeze()  # pyright: ignore[reportAttributeAccessIssue]
        logprob = action_distribution.log_prob(action)
        entropy = action_distribution.entropy()

    return logprob, entropy, value
