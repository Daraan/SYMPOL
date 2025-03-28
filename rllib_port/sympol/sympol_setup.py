from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from ray_utilities.config import ExperimentSetupBase

from ray_utilities.config.create_algorithm import create_algorithm_config

from rllib_port.extended_args import SympolArgumentParser
from rllib_port.sympol.sympol_catalog import SympolJaxPPOCatalog
from rllib_port.sympol.sympol_module import SympolPPOModule

if TYPE_CHECKING:
    from ray_utilities.typing import TrainableReturnData


class SympolSetup(ExperimentSetupBase[SympolArgumentParser]):
    @property
    def project_name(self) -> str:
        return "sympol"

    @property
    def group_name(self) -> str:
        return "sympol"

    def _create_config(self):
        return self.config_from_args(self.args)

    @classmethod
    def config_from_args(cls, args):
        config, _spec = create_algorithm_config(
            args,
            env_type=args.env_type,
            module_class=SympolPPOModule,
            catalog_class=SympolJaxPPOCatalog,
            model_config=args,
            framework="jax",  # cannot use "jax" here
            discrete_eval=False,
        )
        return config

    def create_trainable(self) -> Callable[[dict[str, Any]], TrainableReturnData]:
        def foo(params):
            raise NotImplementedError()

        return foo


if TYPE_CHECKING:
    SympolSetup(None)
