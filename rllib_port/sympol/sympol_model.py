import logging
from typing import TYPE_CHECKING, Tuple
from rllib_port.model_interface import FlaxRLInterface
from sympol import SYMPOL_RL

logger = logging.getLogger(__name__)


class SympolRLModel(SYMPOL_RL, FlaxRLInterface):
    def _forward(self, input_dict: dict, **kwargs) -> dict:
        breakpoint()
        return self.apply(input_dict["obs"])

    def get_num_parameters(self) -> Tuple[int, int]:
        # Unknown
        logger.warning("Requested to know num parameters, but not implemented")
        return 42, 42

    def _set_to_dummy_weights(self, value_sequence=...) -> None:
        # Unknown
        logger.warning("Requested setting to dummy weights, but not implemented")
        return super()._set_to_dummy_weights(value_sequence)


if TYPE_CHECKING:
    SympolRLModel(0, 0, 0, 0, "discrete")
