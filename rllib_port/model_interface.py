import logging
from typing import Tuple

import flax.linen as nn
from ray.rllib.core.models.base import Model

logger = logging.getLogger(__name__)


class FlaxRLInterface(nn.Module, Model):
    def _forward(self, input_dict: dict, **kwargs) -> dict:
        ret = self.apply(input_dict["obs"])
        return ret

    def get_num_parameters(self) -> Tuple[int, int]:
        # Unknown
        logger.warning("Requested to know num parameters, but not implemented")
        return 42, 42

    def _set_to_dummy_weights(self, value_sequence=...) -> None:
        # Unknown
        logger.warning("Requested setting to dummy weights, but not implemented")
        return super()._set_to_dummy_weights(value_sequence)
