from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, cast

import jax
from ray import tune

import configs
from ray_utilities import create_default_trainable
from ray_utilities.config import ExperimentSetupBase
from ray_utilities.config.create_algorithm import create_algorithm_config
from rllib_port.extended_args import SympolArgumentParser
from rllib_port.rllib.connectors.env_to_module import make_env_to_module_without_numpy
from rllib_port.rllib.connectors.module_to_env import make_jax_module_to_env_connector
from rllib_port.rllib.jax_learner import JaxPPOLearner
from rllib_port.sympol.sympol_catalog import SympolJaxPPOCatalog
from rllib_port.sympol.sympol_module import SympolPPOModule

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.algorithms.ppo.ppo import PPOConfig

    from ray_utilities.typing import TrainableReturnData

logger = logging.getLogger(__name__)


class SympolSetup(ExperimentSetupBase[SympolArgumentParser]):
    PROJECT = "SYMPOL"

    @property
    def group_name(self) -> str:
        agent_type = self.args.agent_type
        if agent_type.lower() == "mlp":
            # multiple mlp variants exist
            agent_type = f"{agent_type}({self.PROJECT})"
        return "_".join([agent_type, self.args.env_type, ("-test" if self.args.test else "")])

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

    N_STEPS_DEFAULT = 512

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
        batch_size = args.minibatch_size  # legacy setup has this amount steps per learner pass
        minibatch_size: int = 128  # args.minibatch_size
        while batch_size // minibatch_size < 2:
            minibatch_size = minibatch_size // 2
        return minibatch_size

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
            num_envs_per_env_runner=3,  # env_context.vector_index
            num_env_runners=4,  # env_context.worker_index
            num_cpus_per_env_runner=2,
        )
        # training settings
        # PPO settings
        config.training(
            # LEGACY
            num_epochs=1,  # passes over the batch_size data; handled by update_ppo
        )
        # FIXME: Evaluation currently slow
        config.evaluation(evaluation_interval=20, evaluation_num_env_runners=1)
        # AlgorithmConfig Settings
        logger.info(
            "Setting train_batch_size_per_learner to %s, suggestion %s",
            args.train_batch_size_per_learner,
            cls.get_initial_batch_size(args),
        )
        cast("AlgorithmConfig", config).training(
            add_default_connectors_to_learner_pipeline=True,
            # learner_connector=make_learner_connector_without_numpy(
            #    config, debug=False
            # ),  # ray has wrong annotation here
            learner_class=JaxPPOLearner,
            # This is the size the learner receives per _update
            # Legacy minibatches are done in the learner
            minibatch_size=args.train_batch_size_per_learner,
            train_batch_size_per_learner=args.train_batch_size_per_learner,
            learner_config_dict={
                "rng_key": jax.random.fold_in(jax.random.PRNGKey(args.seed), sum(map(ord, "learner"))),
                "legacy_minibatch_size": args.minibatch_size,
            },
        )
        logger.info(
            "Rllib Minibatch size: %s, Sympol PPO minibatch size suggestion %s",
            args.minibatch_size,
            cls.get_minibatch_size(args),
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

    def create_param_space(self, trial=None) -> dict[str, Any]:
        # FIXME
        param_space_for_tune = super().create_param_space()
        param_space_for_tune["run_seed"] = tune.randint(0, 2**16)
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
