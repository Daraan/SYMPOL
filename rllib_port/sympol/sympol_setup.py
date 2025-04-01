from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import configs
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

    @property
    def model_identifier(self) -> str:
        if self.args.actor == "sympol":
            model_identifier = "-".join([str(self.args.depth), str(self.args.n_estimators), str(self.args.seed)])
        elif self.args.actor != "mlp":
            model_identifier = "-".join([str(self.args.depth), str(self.args.seed)])
        else:
            model_identifier = str(self.args.seed)
        return model_identifier

    def create_parser(self):
        self.parser = SympolArgumentParser()
        return self.parser

    def _create_config(self):
        return self.config_from_args(self.args)

    @classmethod
    def config_from_args(cls, args):
        config, _spec = create_algorithm_config(
            args,
            env_type=args.env_type,
            module_class=SympolPPOModule,
            catalog_class=SympolJaxPPOCatalog,
            model_config=args.as_dict() if hasattr(args, "as_dict") else vars(args).copy(),
            framework="jax",  # cannot use "jax" here
            discrete_eval=False,
        )
        return config

    def create_trainable(self) -> Callable[[dict[str, Any]], TrainableReturnData]:
        def foo(params):
            raise NotImplementedError()

        return foo

    def create_param_space(self, trial):
        # FIXME
        param_space_for_tune = super().create_param_space()
        args = self.args
        if args.actor == "mlp":
            suggested_params = configs.suggest_config_mlp(trial, args.env_id)
        elif args.actor == "sympol":
            suggested_params = configs.suggest_config_sympol(trial, args.env_id)
        elif args.actor == "sdt":
            suggested_params = configs.suggest_config_sdt(trial, args.env_id)
        elif args.actor == "d-sdt":
            suggested_params = configs.suggest_config_dsdt(trial, args.env_id)
        elif args.actor == "stateActionDT":
            suggested_params = configs.suggest_config_stateActionDT(trial, args.env_id)
        else:
            suggested_params = {}
        return suggested_params


if TYPE_CHECKING:
    SympolSetup(None)
