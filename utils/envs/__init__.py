from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
from minigrid.wrappers import OneHotPartialObsWrapper, ViewSizeWrapper

from .random_goal_dist_shift import RandomGoalDistShiftEnv, RandomGoalDistShiftEnv2
from .wrappers import (
    AutoResetWrapper,
    FlatCurrentReducedWrapper,
    FlatCurrentWrapper,
    NormalizeObservationWrapper,
    NormalizeWrapperLunarLander,
)

if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec as _EnvSpec
    from numpy.typing import NDArray

    from args import SympolCLIArgs

__all__ = [
    "AutoResetWrapper",
    "FlatCurrentReducedWrapper",
    "FlatCurrentWrapper",
    "NormalizeObservationWrapper",
    "NormalizeWrapperLunarLander",
    "RandomGoalDistShiftEnv",
    "RandomGoalDistShiftEnv2",
    "build_env",
    "make_training_env",
]


def _make_env(env_id: str | _EnvSpec, *args, **kwargs):
    """Supports creation of deprecated environments that would fail otherwise."""
    try:
        return gym.make(env_id, *args, **kwargs)  # , render_mode="rgb_array")
    except gym.error.DeprecatedEnv as e:
        # Gym reports a warning here
        env_id = str(e).split("Please use ")[-1].split(" instead.")[0].strip(" '`")
        return gym.make(env_id, *args, **kwargs)


def build_env(env_id: str | _EnvSpec, n_env, view_size=3) -> "gym.vector.VectorEnv":
    env: gym.vector.VectorEnv | gym.Env
    if n_env > 1:
        env = _make_env(env_id)
    else:
        env = _make_env(env_id, render_mode="rgb_array")
    if not isinstance(env_id, str):
        env_id = env_id.id

    if "MiniGrid" in env_id:
        env = ViewSizeWrapper(env, agent_view_size=view_size)
        env = OneHotPartialObsWrapper(env)
        env = FlatCurrentReducedWrapper(env)
    elif "LunarLander" in env_id:
        env = NormalizeWrapperLunarLander(env)

    if n_env > 1:
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # Calls os.fork!
        env_to_return = gym.vector.AsyncVectorEnv([lambda env=env: env for _ in range(n_env)])  # type: ignore[arg-type]
    else:
        # still return vector env for consistency
        env_to_return = gym.vector.SyncVectorEnv([lambda e=env: e])
    return env_to_return


def make_training_env(args: SympolCLIArgs) -> tuple[gym.vector.VectorEnv, int, int, list[int]]:
    """
    Returns:
        envs: The vectorized environment
        obs_dim: The dimension of the observation space
        action_dim: The dimension of the action space
        action_indices: The indices of the actions
    """
    envs = build_env(args.env_id, n_env=args.n_envs, view_size=args.view_size)

    obs_dim = envs.single_observation_space.shape[-1]  # type: ignore

    print("Observations:", obs_dim)
    if isinstance(envs.single_action_space, gym.spaces.Discrete):
        if any(
            substring in args.env_id
            for substring in ["Crossing", "DistShift", "Empty", "LavaGap", "FourRooms", "Dynamic-Obstacles"]
        ):
            action_dim = 3
            action_indices = [0, 1, 2]
        elif any(substring in args.env_id for substring in ["MultiRoom", "Unlock", "GoToDoor", "RedBlueDoors"]):
            action_dim = 4
            if "GoToDoor" in args.env_id:
                action_indices = [0, 1, 2, 6]
            else:
                action_indices = [0, 1, 2, 5]
        elif any(substring in args.env_id for substring in ["UnlockPickup", "DoorKey"]):
            action_dim = 5
            action_indices = [0, 1, 2, 3, 5]
        # elif any(substring in args.env_id for substring in ['forex']):
        #    action_dim = 2
        else:
            action_dim = int(envs.single_action_space.n)
            action_indices = list(range(action_dim))
        print("Actions:", action_dim)
    elif isinstance(envs.single_action_space, gym.spaces.Box):
        action_dim = envs.single_action_space.shape[-1]
        action_indices = list(range(action_dim))
        print("Actions:", action_dim)
    else:
        raise NotImplementedError(f"Action space type '{type(envs.single_action_space)}' not implemented")
    return envs, obs_dim, action_dim, action_indices


def _select_action(env_id: str, action: int | NDArray, action_indices: list[int]):
    if any(
        substring in env_id
        for substring in [
            "MultiRoom",
            "Unlock",
            "GoToDoor",
            "UnlockPickup",
            "DoorKey",
            "RedBlueDoors",
        ]
    ):
        action = action_indices[action]
    if np.ndim(action) == 0:
        action = np.array([action])
    return action
