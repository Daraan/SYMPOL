from __future__ import annotations

from typing import TYPE_CHECKING

from ray_utilities.config import ExperimentSetupBase

from ray_utilities.config.create_algorithm import create_algorithm_config

from rllib_port.extended_args import SympolArgumentParser
from rllib_port.sympol.sympol_catalog import SympolPPOCatalog
from rllib_port.sympol.sympol_module import SympolPPOModule

if TYPE_CHECKING:

    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig


class SympolSetup(ExperimentSetupBase[SympolArgumentParser]):
    @property
    def project_name(self) -> str:
        return "sympol"

    @property
    def group_name(self) -> str:
        return "sympol"

    def _create_config(self) -> AlgorithmConfig:
        return self.config_from_args(self.args)

    @classmethod
    def config_from_args(cls, args) -> AlgorithmConfig:
        config, _spec = create_algorithm_config(
            args,
            env_type=args.env_type,
            module_class=SympolPPOModule,
            catalog_class=SympolPPOCatalog,
            model_config=args,
            framework="jax",  # cannot use "jax" here
            discrete_eval=False,
        )
        return config


if TYPE_CHECKING:
    SympolSetup()
