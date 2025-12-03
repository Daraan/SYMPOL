from typing import Any
from ray.rllib.algorithms import AlgorithmConfig, PPOConfig

from sympol.config_types.params_types import SympolCatalogOptions


class SympolAlgorithmConfig(AlgorithmConfig):
    @property
    def _model_config_auto_includes(self) -> dict[str, Any]:
        """Defines which `AlgorithmConfig` settings/properties should be
        auto-included into `self.model_config`.

        The dictionary in this property contains the default configuration of an
        algorithm. Together with the `self._model`, this method is used to
        define the configuration sent to the `RLModule`.

        Returns:
            A dictionary with the automatically included properties/settings of this
            `AlgorithmConfig` object into `self.model_config`.
        """
        return super()._model_config_auto_includes | {
            k: getattr(self, k) for k in SympolCatalogOptions.__annotations__.keys() if hasattr(self, k)
        }


class SympolPPOConfig(PPOConfig, SympolAlgorithmConfig):
    pass
