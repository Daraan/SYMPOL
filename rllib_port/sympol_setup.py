from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, cast
from typing_extensions import deprecated

import jax
from ray import tune

import configs
from ray_utilities.config.create_algorithm import create_algorithm_config
from ray_utilities.config.experiment_base import ExperimentSetupBase
from ray_utilities.config.extensions import SetupWithDynamicBuffer
from ray_utilities.default_trainable import create_default_trainable
from rllib_port.core.connectors.env_to_module import make_env_to_module_without_numpy
from rllib_port.core.connectors.module_to_env import make_jax_module_to_env_connector
from rllib_port.core.jax_learner import JaxPPOLearner
from rllib_port.core.sympol_catalog import SympolJaxPPOCatalog
from rllib_port.core.sympol_module import SympolPPOModule
from rllib_port.extended_args import SympolArgumentParser

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.algorithms.ppo.ppo import PPOConfig

    from ray_utilities.typing import TrainableReturnData

logger = logging.getLogger(__name__)


class SympolSetup(SetupWithDynamicBuffer, ExperimentSetupBase[SympolArgumentParser]):
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
    @deprecated("legacy")
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

    @classmethod
    def apply_legacy_settings(cls, config: AlgorithmConfig | PPOConfig, args: SympolArgumentParser | Any) -> None:
        assert args.legacy
        config.training(
            num_epochs=1,  # passes over the batch_size data; handled by update_ppo
            learner_config_dict={"legacy_minibatch_size": args.minibatch_size, "legacy": True},
            # Minibatching is done within the learner, EnvRunners should create new rollout every time.
            minibatch_size=args.train_batch_size_per_learner,
            train_batch_size_per_learner=args.train_batch_size_per_learner,
        )
        if args.minibatch_size > args.train_batch_size_per_learner:
            logger.error(
                "Minibatch size (%s) is larger than train batch size (%s). Likely leads to errors.",
                args.minibatch_size,
                args.train_batch_size_per_learner,
            )
        if args.train_batch_size_per_learner % args.minibatch_size != 0:
            logger.warning(
                "Train batch size (%s) is not divisible by minibatch size (%s). This may lead to errors.",
                args.train_batch_size_per_learner,
                args.minibatch_size,
            )

    @classmethod
    def _config_from_args(cls, args):
        config, _spec = create_algorithm_config(
            args,
            env_type=args.env_type,
            module_class=SympolPPOModule,
            catalog_class=SympolJaxPPOCatalog,
            model_config=args.as_dict() if hasattr(args, "as_dict") else vars(args).copy(),
            framework="torch",  # cannot use "jax" here
            discrete_eval=False,
        )
        cls.add_callbacks_to_config(config, cls._get_callbacks_from_args(args))
        # NOTE: Incomplete automcompletion kwargs -> super()
        # Algorithm settings
        DEBUG_CONNECTORS = {"env_to_module": False, "module_to_env": False, "learner": False}
        config.env_runners(
            # env -> module
            add_default_connectors_to_env_to_module_pipeline=False,
            env_to_module_connector=make_env_to_module_without_numpy(config, debug=DEBUG_CONNECTORS["env_to_module"]),
            # module -> env
            add_default_connectors_to_module_to_env_pipeline=False,
            module_to_env_connector=make_jax_module_to_env_connector(
                config,
                key=jax.random.fold_in(jax.random.PRNGKey(args.seed), sum(map(ord, "module_to_env_connector"))),
                debug=DEBUG_CONNECTORS["module_to_env"],
            ),
            # TODO: Should set this in the defaults of the submodule
            num_envs_per_env_runner=3,  # env_context.vector_index
            num_env_runners=4 if args.parallel else 1,  # env_context.worker_index
            num_cpus_per_env_runner=2 if args.parallel else 1,
        )
        # training settings
        cast("AlgorithmConfig", config).training(
            add_default_connectors_to_learner_pipeline=True,
            # learner_connector=make_learner_connector_without_numpy(
            #    config, debug=False
            # ),  # ray has wrong annotation here
            learner_class=JaxPPOLearner,
            # This is the size the learner receives per _update
            # Legacy minibatches are done in the learner
            learner_config_dict={
                "rng_key": jax.random.fold_in(jax.random.PRNGKey(args.seed), sum(map(ord, "learner"))),
                "accumulate_gradients_every": args.accumulate_gradients_every,
                "dynamic_buffer": args.dynamic_buffer,
                "dynamic_batch": not args.static_batch,
                "legacy": args.legacy,
                "_debug_connectors": DEBUG_CONNECTORS["learner"],  # Not Implemented yet
            },
        )
        # PPO specific training settings
        # config.training()
        if args.legacy:
            logger.debug("Using legacy implementation")
            cls.apply_legacy_settings(config, args)
        else:
            config.training(
                num_epochs=20,  # passes over the batch_size data; handled by update_ppo
                minibatch_size=args.minibatch_size,
                train_batch_size_per_learner=args.train_batch_size_per_learner,
            )
            assert not config.learner_config_dict.get("legacy")
        # logging
        logger.info(
            "Rllib Minibatch size: %s, Sympol PPO minibatch size suggestion %s",
            args.minibatch_size,
            cls.get_minibatch_size(args),
        )
        if args.seed is None:
            logger.warning("Args seed is None")
            args.seed = SympolArgumentParser.seed
        return config

    def create_trainable(self) -> Callable[[dict[str, Any]], TrainableReturnData]:
        return create_default_trainable(setup=self, setup_class=type(self))

    def create_param_space(self, trial=None) -> dict[str, Any]:
        # FIXME use tune
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
