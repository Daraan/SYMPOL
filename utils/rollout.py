from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol
from typing_extensions import Unpack, TypeAliasType

import jax.numpy as jnp

from utils.get_action_and_value import get_action_and_value
import numpy as np

if TYPE_CHECKING:
    from ray_utilities.jax.jax_model import PureJaxModelProtocol
    import chex
    from numpy.typing import NDArray

    from config_types.args_types import CLIArgs
    from mlp import Actor_MLP, Actor_MLP_Continuous, Critic_MLP
    from sdt import Actor_SDT, Critic_SDT
    from sympol import SYMPOL_RL
    from utils.utils import ActorTrainState, EpisodeStatistics, Storage, TrainState

    _Actor = Actor_MLP | Actor_MLP_Continuous | Actor_SDT | SYMPOL_RL | PureJaxModelProtocol
    _Critic = Critic_MLP | Critic_SDT


class UpdateNStepsArgs(Protocol):
    total_steps: int
    n_envs: int
    dynamic_buffer: bool
    static_batch: bool


def update_buffer_and_rollout_size(
    args: UpdateNStepsArgs,
    *,
    initial_steps: int,
    global_step: int,
    accumulate_gradients_every_initial: int,
):
    """
    Calculates a new rollout and batch size

    Afterwards create Rollout with `n_steps`
    `if args.dynamic_buffer or not args.static_batch:` recalculate
    Then if n_steps != n_steps_old: -> create rollout

    If `args.dynamic_buffer` is True, the buffer size `n_steps` is increased.
    Otherwise `n_steps == initial_steps`.
    Otherwise only `batch_size` and `accumulate_gradients_every` are increased.
    """
    # increase_index = global_step // (args.total_steps//sum(increase_factor_list))
    increase_factor = int(
        2 ** (np.ceil((((global_step + 1) * 8) / (1 + args.total_steps))) - 1)
    )  # int(increase_factor_list_long[increase_index])
    increase_factor_batch = int(
        2 ** (np.ceil((((global_step + 1) * 8) / (1 + args.total_steps))) - 1)
    )  # int(increase_factor_list_long[increase_index])
    if args.dynamic_buffer:
        n_steps = initial_steps * increase_factor
    else:
        n_steps = initial_steps
    if not args.static_batch:
        accumulate_gradients_every = int(accumulate_gradients_every_initial * increase_factor_batch)
    else:
        accumulate_gradients_every = int(accumulate_gradients_every_initial)
    # DYNAMIC_BATCH_SIZE
    batch_size = int(args.n_envs * n_steps)  # XXX: Get rid of n_envs; samples_per_step
    # n_iterations = args.total_steps // batch_size
    # eval_freq = max(args.eval_freq // batch_size, 1)
    return batch_size, accumulate_gradients_every, n_steps


_RolloutSignature = TypeAliasType(
    "_RolloutSignature",
    """tuple[
    ActorTrainState, TrainState, EpisodeStatistics, NDArray, NDArray[np.bool_], Storage, chex.PRNGKey, int
]""",
)
RolloutCallableType = TypeAliasType("RolloutCallableType", Callable[[Unpack[_RolloutSignature]], _RolloutSignature])


def create_rollout_function(
    n_steps, envs, *, args: CLIArgs, actor: _Actor, critic: _Critic, action_indices: list[int]
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
            next_obs, reward, next_done, trunc, info = envs.step(action)
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
