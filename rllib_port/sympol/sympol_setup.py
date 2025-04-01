from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, cast

import jax
from ray_utilities import create_default_trainable

import configs
from ray_utilities.config import ExperimentSetupBase
from ray_utilities.config.create_algorithm import create_algorithm_config
from rllib_port.connectors.env_to_module import make_env_to_module_without_numpy
from rllib_port.connectors.module_to_env import make_jax_module_to_env_connector
from rllib_port.extended_args import SympolArgumentParser
from rllib_port.jax_learner import JaxPPOLearner
from rllib_port.sympol.sympol_catalog import SympolJaxPPOCatalog
from rllib_port.sympol.sympol_module import SympolPPOModule
from gymnasium.envs.registration import VectorizeMode

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.algorithms.ppo.ppo import PPOConfig
    from ray_utilities.typing import TrainableReturnData

logger = logging.getLogger(__name__)


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

    @staticmethod
    def get_minibatch_size(args):
        # if trial sample this
        if False and trial:
            ...
        elif not args.use_best_config:
            n_steps = SympolSetup.N_STEPS_DEFAULT
            n_envs = args.n_envs  # HACK; modify postprocessing args
        else:
            n_envs = args.n_envs
            n_steps = args.n_steps
        if args.dynamic_buffer:
            n_steps = max(16, n_steps // 8)
        batch_size = int(n_envs * n_steps)
        minibatch_size: int = args.minibatch_size
        while batch_size // minibatch_size < 2:
            minibatch_size = minibatch_size // 2
        return minibatch_size

    N_STEPS_DEFAULT = 128

    @staticmethod
    def get_initial_batch_size(args):
        if False and trial:
            ...
        elif not args.use_best_config:
            n_steps = SympolSetup.N_STEPS_DEFAULT
            n_envs = args.n_envs
        else:
            n_steps = args.n_steps
            n_envs = args.n_envs
        if args.dynamic_buffer:
            n_steps = max(16, n_steps // 8)

        batch_size = int(n_envs * n_steps)
        return batch_size

    @classmethod
    def config_from_args(cls, args):
        config, _spec = create_algorithm_config(
            args,
            env_type=args.env_type,
            module_class=SympolPPOModule,
            catalog_class=SympolJaxPPOCatalog,
            model_config=args.as_dict() if hasattr(args, "as_dict") else vars(args).copy(),
            framework="torch",  # cannot use "jax" here
            discrete_eval=False,
        )
        # NOTE: Incomplete automcompletion kwargs -> super()
        # Algorithm settings
        DEBUG_CONNECTORS = True
        if args.seed is None:
            args.seed = SympolArgumentParser.seed
        config.env_runners(
            add_default_connectors_to_env_to_module_pipeline=False,
            env_to_module_connector=make_env_to_module_without_numpy(config, debug=False),
            add_default_connectors_to_module_to_env_pipeline=False,
            module_to_env_connector=make_jax_module_to_env_connector(
                config,
                key=jax.random.fold_in(jax.random.PRNGKey(args.seed), sum(map(ord, "module_to_env_connector"))),
                debug=False,
            ),
            # experimental
            # gym_env_vectorize_mode=VectorizeMode.ASYNC,
        )
        # training settings
        # PPO settings
        config.training()
        # AlgorithmConfig Settings
        logger.info("Setting train_batch_size_per_learner to %s", cls.get_initial_batch_size(args))
        print("Setting train_batch_size_per_learner to %s", cls.get_initial_batch_size(args))
        cast("AlgorithmConfig", config).training(
            add_default_connectors_to_learner_pipeline=True,
            # learner_connector=make_learner_connector_without_numpy(
            #    config, debug=False
            # ),  # ray has wrong annotation here
            learner_class=JaxPPOLearner,
            minibatch_size=cls.get_minibatch_size(args),
            train_batch_size_per_learner=cls.get_initial_batch_size(args),
            learner_config_dict={
                "rng_key": jax.random.fold_in(jax.random.PRNGKey(args.seed), sum(map(ord, "learner")))
            },
        )
        if 0:
            cast("AlgorithmConfig", config).training(
                learner_config_dict={},
                optimizer=...,
                add_default_connectors_to_learner_pipeline=True,  # maybe false
            )
        return config

    def create_trainable(self) -> Callable[[dict[str, Any]], TrainableReturnData]:
        return create_default_trainable(setup=self, setup_class=type(self))

    def create_param_space(self, trial=None):
        # FIXME
        param_space_for_tune = super().create_param_space()
        if not trial:
            return param_space_for_tune
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
