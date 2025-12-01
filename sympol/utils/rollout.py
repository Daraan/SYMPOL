from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol

import jax.numpy as jnp
import numpy as np
from typing_extensions import TypeAliasType, Unpack

from sympol.utils.get_action_and_value import get_action_and_value

if TYPE_CHECKING:
    import chex
    import gymnasium as gym
    from numpy.typing import NDArray

    from sympol.config_types.args_types import SympolCLIArgs
    from sympol.mlp import Actor_MLP, Actor_MLP_Continuous, Critic_MLP
    from ray_utilities.jax.jax_model import PureJaxModelProtocol
    from sympol.sdt import Actor_SDT, Critic_SDT
    from sympol.sympol import SYMPOL_RL
    from sympol.utils.utils import ActorTrainState, EpisodeStatistics, Storage, TrainState

    _Actor = Actor_MLP | Actor_MLP_Continuous | Actor_SDT | SYMPOL_RL | PureJaxModelProtocol
    _Critic = Critic_MLP | Critic_SDT


# NOTE: This is a copy of the function in ray_utilities for ppo_new_interface to be standalone
# However, this function does not assure that the return values are bounded in sensible ranges.
# NOTE: SYMPOL keeps a copy of this function in the repo (standalone)
def update_buffer_and_rollout_size(
    *,
    total_steps: int,
    dynamic_buffer: bool,
    dynamic_batch: bool,
    n_envs: int = 1,
    initial_steps: int,
    global_step: int,
    num_increase_factors: int = 8,
    accumulate_gradients_every_initial: int,
):
    """
    Calculates a new rollout and batch size

    Afterwards create Rollout with `n_steps`
    `if args.dynamic_buffer or not args.static_batch:` recalculate
    Then if n_steps != n_steps_old: -> create rollout

    Attention:
        The default value of num_increase_factors=8, does not match with the default values of
        min_size=32 and max_size=8192 in the other functions of this module, as those result in
        9 different values; use num_increase_factors=9 to match the other functions.
    """
    # increase_index = global_step // (args.total_steps//sum(increase_factor_list))
    if global_step + 1 > total_steps:
        global_step = total_steps  # prevent explosion; limit factor to 128
    increase_factor = int(
        2 ** (np.ceil((((global_step + 1) * num_increase_factors) / (1 + total_steps))) - 1)
    )  # int(increase_factor_list_long[increase_index])
    increase_factor_batch = int(
        2 ** (np.ceil((((global_step + 1) * num_increase_factors) / (1 + total_steps))) - 1)
    )  # int(increase_factor_list_long[increase_index])
    if dynamic_buffer:
        n_steps = initial_steps * increase_factor
    else:
        n_steps = initial_steps
    if dynamic_batch:
        accumulate_gradients_every = int(accumulate_gradients_every_initial * increase_factor_batch)
    else:
        accumulate_gradients_every = int(accumulate_gradients_every_initial)
    # DYNAMIC_BATCH_SIZE
    batch_size = int(n_envs * n_steps)  # XXX: Get rid of n_envs; samples_per_step
    # n_iterations = args.total_steps // batch_size
    # eval_freq = max(args.eval_freq // batch_size, 1)
    # logger.debug("updating buffer after step %d / %s to %s. Initial size: %s", global_step, args.total_steps, batch_size, initial_steps)

    return batch_size, accumulate_gradients_every, n_steps


_RolloutSignature = TypeAliasType(
    "_RolloutSignature",
    """tuple[
    ActorTrainState, TrainState, EpisodeStatistics, NDArray, NDArray[np.bool_], Storage, chex.PRNGKey, int
]""",
)
RolloutCallableType = TypeAliasType("RolloutCallableType", Callable[[Unpack[_RolloutSignature]], _RolloutSignature])


def create_rollout_function(
    n_steps: int,
    envs: gym.vector.VectorEnv,
    *,
    args: SympolCLIArgs,
    actor: _Actor,
    critic: _Critic,
    action_indices: list[int],
) -> RolloutCallableType:
    def rollout_(
        actor_state: ActorTrainState,
        critic_state: TrainState,
        episode_stats: EpisodeStatistics,
        next_obs: NDArray,
        next_done: NDArray[np.bool_],
        storage: Storage,
        key: chex.PRNGKey,
        global_step: int,
    ) -> _RolloutSignature:
        for step in range(n_steps):
            global_step += args.n_envs
            storage, action, key = get_action_and_value(
                actor_state.params,
                critic_state,
                next_obs,
                next_done,
                storage,
                step,
                key,
                action_type=args.action_type,
                actor=actor,
                critic=critic,
                actor_state_indices=actor_state.indices,
            )
            # TRY NOT TO MODIFY: execute the game and log data.
            action = np.array(action)
            if any(
                substring in args.env_id
                for substring in ["MultiRoom", "Unlock", "GoToDoor", "UnlockPickup", "DoorKey", "RedBlueDoors"]
            ):
                action = np.array([action_indices[single_action] for single_action in action])
            next_obs, reward, next_done, trunc, _info = envs.step(action)
            new_episode_return = episode_stats.episode_returns + reward
            new_episode_length = episode_stats.episode_lengths + 1
            episode_stats = episode_stats.replace(
                episode_returns=(new_episode_return) * (1 - next_done) * (1 - trunc),
                episode_lengths=(new_episode_length) * (1 - next_done) * (1 - trunc),
                # only update the `returned_episode_returns` if the episode is done
                returned_episode_returns=jnp.where(
                    next_done + trunc,
                    new_episode_return,
                    episode_stats.returned_episode_returns,
                ),
                returned_episode_lengths=jnp.where(
                    next_done + trunc,
                    new_episode_length,
                    episode_stats.returned_episode_lengths,
                ),
            )
            storage = storage.replace(rewards=storage.rewards.at[step].set(reward))
        return actor_state, critic_state, episode_stats, next_obs, next_done, storage, key, global_step

    return rollout_
