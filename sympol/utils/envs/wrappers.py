from __future__ import annotations

import functools
import operator
import random
from functools import reduce
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObservationWrapper

if TYPE_CHECKING:
    import chex

try:
    from gymnax.environments import environment as environment_gymnax  # pyright: ignore[reportMissingImports]
    from gymnax.environments import spaces as spaces_gymnax  # pyright: ignore[reportMissingImports]
    from gymnax.wrappers.purerl import GymnaxWrapper  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
    print("Gymnax not installed; skipping related wrappers")

__all__ = [
    "AutoResetWrapper",
    "FlatCurrentReducedWrapper",
    "FlatCurrentWrapper",
    "NormalizeObservationWrapper",
    "NormalizeWrapperLunarLander",
]


class AutoResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super(AutoResetWrapper, self).__init__(env)
        self.reset_on_step = False
        self.reset_env()

    def reset_env(self):
        # Generate a new random seed
        seed = random.randint(0, 1000000)
        self.observation, self.info = self.env.reset(seed=seed)

    def step(self, action):
        if self.reset_on_step:
            self.reset_env()
            self.reset_on_step = False

        observation, reward, done, truncated, info = self.env.step(action)
        if done or truncated:
            self.reset_on_step = True
        return observation, reward, done, truncated, info

    def reset(self, **kwargs):
        self.reset_env(**kwargs)
        return self.observation, self.info


class NormalizeWrapperLunarLander(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        observation[0] = (observation[0] - 0) / 1.5
        observation[1] = (observation[1] - 0) / 1.5
        observation[2] = (observation[2] - 0) / 5.0
        observation[3] = (observation[3] - 0) / 5.0
        observation[4] = (observation[4] - 0) / 3.14
        observation[5] = (observation[5] - 0) / 5.0
        observation[6] = (observation[6] - 1) / 0.5
        observation[7] = (observation[7] - 1) / 0.5

        return observation


class FlatCurrentReducedWrapper(ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array.

    This wrapper is not applicable to BabyAI environments, given that these have their own language component.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import FlatObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> env_obs = FlatObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs.shape
        (2835,)
    """

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        imgSpace = env.observation_space.spaces["image"]

        self.select_indices = [0, 1, 2, 8, 9]
        # Define a mapping from environment names to select indices
        env_select_indices = {
            "DistShift": [0, 1, 2, 8, 9],  # left, right, forward
            "LavaGap": [0, 1, 2, 8, 9],  # left, right, forward
            "LavaCrossing": [0, 1, 2, 8, 9],  # left, right, forward
            "SimpleCrossing": [0, 1, 2, 8],  # left, right, forward
            "FourRooms": [0, 1, 2, 8],  # left, right, forward
            "Empty": [0, 1, 2, 8],  # left, right, forward
            "MultiRoom": [0, 1, 2, 4, 8, 17, 18],  # left, right, forward, toggle
            "Dynamic-Obstacles": [0, 1, 2, 4, 6, 8],  # left, right, forward
            "Unlock": [0, 1, 2, 4, 5, 8, 17, 18, 19],  # left, right, forward, toggle #No pickup key
            "UnlockPickup": [0, 1, 2, 4, 5, 7, 8, 17, 18, 19],  # left, right, forward, pickup, toggle #No pickup key
            "DoorKey": [0, 1, 2, 4, 5, 8, 17, 18, 19],  # left, right, forward, pickup, toggle #Pickup key
            "GoToDoor": [0, 1, 2, 4, 8, 11, 12, 13, 14, 15, 16],  # left, right, forward, done
            "RedBlueDoors": [0, 1, 2, 4, 8, 11, 13, 17, 18],  # left, right, forward, toggle
            "PutNear": [0, 1, 2, 4, 8, 17, 18],  # left, right, forward, pickup, drop
        }

        # Get the environment name
        env_name = env.spec.id

        env_identifier = env_name
        for key in env_select_indices.keys():
            if key in env_name:
                env_identifier = key

        # Set select_indices based on the environment name
        if env_identifier in env_select_indices:
            self.select_indices = env_select_indices[env_identifier]
            print(f"Environment {env_identifier} with Observations {self.select_indices}")
        else:
            raise ValueError(f"Environment {env_identifier} is not supported by this wrapper.")

        imgSize = (
            imgSpace.shape[0] * imgSpace.shape[1] * len(self.select_indices)
        )  # reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize,),
            dtype=np.float32,
        )

        self.cachedStr: str = None  # type: ignore

    def observation(self, observation):
        image = observation["image"]
        mission = observation["mission"]
        # print('image.shape', image.shape)
        # print('image.flatten().shape', image.flatten().shape)
        obs = image[:, :, self.select_indices].flatten().astype(np.float32)
        obs = obs * 2 - 1  # convert to range -1,1 instead of 0,1

        # obs =
        # print('obs.shape', obs.shape)
        return obs


class FlatCurrentWrapper(ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array.

    This wrapper is not applicable to BabyAI environments, given that these have their own language component.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import FlatObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> env_obs = FlatObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs.shape
        (2835,)
    """

    def __init__(self, env, maxStrLen=96):
        super().__init__(env)

        imgSpace = self.env.observation_space.spaces["image"]  # type: ignore # add better generics
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize,),
            dtype=np.float32,
        )

        self.cachedStr: str = None  # type: ignore

    def observation(self, observation):
        image = observation["image"]
        mission = observation["mission"]

        obs = image.flatten().astype(np.float32)
        obs = obs * 2 - 1  # convert to range -1,1 instead of 0,1
        return obs


try:
    from gymnax.environments import environment as environment_gymnax
    from gymnax.environments import spaces as spaces_gymnax
    from gymnax.wrappers.purerl import GymnaxWrapper
except ModuleNotFoundError:
    print("Gymnax not installed; skipping related wrappers")

    if not TYPE_CHECKING:

        class NormalizeObservationWrapper:
            """Normalize the observations of the environment."""

            def __init__(self, env: environment_gymnax.Environment, params):
                raise ModuleNotFoundError("Gymnax not installed; skipping related wrappers")

else:

    class NormalizeObservationWrapper(GymnaxWrapper):
        """Normalize the observations of the environment."""

        def __init__(self, env: environment_gymnax.Environment, params):
            super().__init__(env)
            self._env: environment_gymnax.Environment

            self.original_low_no_clip = self._env.observation_space(params).low
            self.original_high_no_clip = self._env.observation_space(params).high
            self.original_low = jnp.clip(self._env.observation_space(params).low, -10, 10)
            self.original_high = jnp.clip(self._env.observation_space(params).high, -10, 10)

        def observation_space(self, params) -> spaces_gymnax.Box:
            assert isinstance(self._env.observation_space(params), spaces_gymnax.Box), (
                "Only Box spaces are supported for now."
            )

            space = spaces_gymnax.Box(
                low=-0.5 + (self.original_low_no_clip - self.original_low) / (self.original_high - self.original_low),
                high=-0.5 + (self.original_high_no_clip - self.original_low) / (self.original_high - self.original_low),
                shape=self._env.observation_space(params).shape,
                dtype=self._env.observation_space(params).dtype,
            )
            print(
                -0.5 + (self.original_low_no_clip - self.original_low) / (self.original_high - self.original_low),
                self.original_high,
                self.original_low,
                self.original_low_no_clip,
            )
            print(space.low, space.high)
            return space

        def normalize_obs(self, obs: jnp.ndarray) -> jnp.ndarray:
            return -0.5 + (obs - self.original_low) / (self.original_high - self.original_low)

        @functools.partial(jax.jit, static_argnums=(0,))
        def reset(
            self, key: chex.PRNGKey, params: Optional[environment_gymnax.EnvParams] = None
        ) -> Tuple[chex.Array, environment_gymnax.EnvState]:
            obs, state = self._env.reset(key, params)
            obs = self.normalize_obs(obs)  # jnp.reshape(obs, (-1,))
            return obs, state

        @functools.partial(jax.jit, static_argnums=(0,))
        def step(
            self,
            key: chex.PRNGKey,
            state: environment_gymnax.EnvState,
            action: Union[int, float],  # noqa: PYI041
            params: Optional[environment_gymnax.EnvParams] = None,
        ) -> Tuple[chex.Array, environment_gymnax.EnvState, float, bool, Any]:  # dict]:
            obs, state, reward, done, info = self._env.step(key, state, action, params)
            obs = self.normalize_obs(obs)  # jnp.reshape(obs, (-1,))
            return obs, state, reward, done, info
