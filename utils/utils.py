from __future__ import annotations

from typing import TYPE_CHECKING

import flax
import flax.struct
from flax.training.train_state import TrainState
from typing_extensions import Self, TypeVar

from ray_utilities.jax.utils import ExtendedTrainState
from utils.envs import build_env, make_training_env
from utils.trees import (
    convert_to_child_representation,
    convert_to_child_representation_soft,
    convert_to_discrete_tree,
    count_nodes,
    plot_decision_tree,
    plot_decision_tree_soft,
    plot_tree_from_representation,
    plot_tree_from_representation_soft,
    prune_and_merge_tree,
)

if TYPE_CHECKING:
    import jax.numpy as jnp

__all__ = [
    "OBSERVATION_LABELS",
    "ActorTrainState",
    "EpisodeStatistics",
    "ObservationActionBuffer",
    "Storage",
    "TrainState",  # re export
    "build_env",
    "convert_to_child_representation",
    "convert_to_child_representation_soft",
    "convert_to_discrete_tree",
    "count_nodes",
    "make_training_env",
    "plot_decision_tree",
    "plot_decision_tree_soft",
    "plot_tree_from_representation",
    "plot_tree_from_representation_soft",
    "prune_and_merge_tree",
]

_is_discreteT = TypeVar("_is_discreteT", bound=bool, default=bool)  # noqa: N816, PYI018
"""Generic to be used with models classes to better infer their return type depending on the action space"""

OBSERVATION_LABELS = {
    "LunarLander-v2": [
        "x",
        "y",
        "velocity_x",
        "velocity_y",
        "angle",
        "angular_velocity",
        "leg_1_ground_contact",
        "leg_2_ground_contact",
    ]
}


class ActorTrainState(ExtendedTrainState):
    indices: dict = flax.struct.field(pytree_node=False, hash=False)
    # TODO:
    # possibly use: core.FrozenDict[str, Any] = struct.field(pytree_node=True)


def format_array(arr) -> str:
    return f"{arr[:2]}\n ...\n{arr[-2:]}" if arr.size > 4 else str(arr)


if TYPE_CHECKING:

    class _HasReplaceMixing:
        def replace(self, *args, **kwargs) -> Self: ...
else:
    _HasReplaceMixing = object


@flax.struct.dataclass(kw_only=True)
class StorageNoValues(_HasReplaceMixing):
    obs: jnp.ndarray
    actions: jnp.ndarray
    logprobs: jnp.ndarray
    dones: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray
    rewards: jnp.ndarray

    def __str__(self) -> str:
        try:
            return (
                f"Storage(\n"
                f"obs {self.obs.shape}={format_array(self.obs)},\n"
                f"actions {self.actions.shape}={format_array(self.actions)},\n"
                f"logprobs {self.logprobs.shape}={format_array(self.logprobs)},\n"
                f"dones {self.dones.shape}={format_array(self.dones)},\n"
                f"values {self.values.shape}={format_array(self.values)},\n" if hasattr(self, "values") else "" # pyright: ignore[reportAttributeAccessIssue]
                f"advantages {self.advantages.shape}={format_array(self.advantages)},\n"
                f"returns {self.returns.shape}={format_array(self.returns)},\n"
                f"rewards {self.rewards.shape}={format_array(self.rewards)}\n"
                f")"
            )  # fmt: skip
        except Exception as e:  # noqa: BLE001
            return f"Error in __repr__: {e!s}" + super().__repr__()

    def __repr__(self) -> str:
        try:
            return (
                f"Storage(\n"
                f"obs={format_array(self.obs)},\n"
                f"actions={format_array(self.actions)},\n"
                f"logprobs={format_array(self.logprobs)},\n"
                f"dones={format_array(self.dones)},\n"
                f"values={format_array(self.values)},\n" if hasattr(self, "values") else ""  # pyright: ignore[reportAttributeAccessIssue]
                f"advantages={format_array(self.advantages)},\n"
                f"returns={format_array(self.returns)},\n"
                f"rewards={format_array(self.rewards)}\n"
                f")"
            )  # fmt: skip
        except Exception as e:  # noqa: BLE001
            return f"Error in __repr__: {e!s}" + super().__repr__()


@flax.struct.dataclass(kw_only=True)
class Storage(StorageNoValues):
    values: jnp.ndarray


@flax.struct.dataclass
class EpisodeStatistics(_HasReplaceMixing):
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray
    returned_episode_returns: jnp.ndarray
    returned_episode_lengths: jnp.ndarray

    if TYPE_CHECKING:

        def replace(self, *args, **kwargs) -> Self: ...


@flax.struct.dataclass
class ObservationActionBuffer(_HasReplaceMixing):
    obs: jnp.ndarray
    actions: jnp.ndarray
