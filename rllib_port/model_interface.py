from __future__ import annotations
from abc import abstractmethod
import logging

import jax
from ray.rllib.core.models.base import Model
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import flax.linen as nn
    from utils.utils import ActorTrainState, TrainState
    import chex
    from ray.rllib.utils.typing import TensorType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # want nn interface but do not inherit from it
    class __Parent(nn.Module, Model): ...
else:
    __Parent = Model


class FlaxRLModel(__Parent):
    @abstractmethod
    def init_state(self, envs, actor_key: chex.PRNGKey) -> ActorTrainState | TrainState:
        pass

    def _forward(self, input_dict: dict, **kwargs) -> dict | TensorType:
        variables, indices = "XXX", "XXX"  # TODO: implement
        return self.apply(inputs=input_dict["obs"], **kwargs)

    def __call__(self, *args, **kwargs):
        # This is a dummy method to do checked forward passes.
        return self._forward(*args, **kwargs)

    def get_num_parameters(self) -> tuple[int, int]:
        # Unknown
        logger.warning("Warning num_parameters called which might be wrong")
        try:
            param_count = sum(x.size for x in jax.tree_util.tree_leaves(self))
            return (
                param_count,  # trainable
                param_count - param_count,  # non trainable? 0?
            )
        except Exception:
            logger.exception("Error getting number of parameters")
            return 42, 42

    def _set_to_dummy_weights(self, value_sequence=...) -> None:
        # Unknown
        logger.warning("Requested setting to dummy weights, but not implemented")
        return super()._set_to_dummy_weights(value_sequence)
