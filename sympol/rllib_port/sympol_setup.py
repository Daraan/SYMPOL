from __future__ import annotations

import logging
import sys
from inspect import ismethod
from typing import TYPE_CHECKING, Any, cast

import jax
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from typing_extensions import NotRequired, deprecated
from typing_extensions import get_origin as type_get_origin

from ray_utilities.config import add_callbacks_to_config
from ray_utilities.config.create_algorithm import create_algorithm_config
from ray_utilities.connectors.jax.env_to_module import EnvToModuleWithoutNumpyConnector
from ray_utilities.connectors.jax.module_to_env import MakeJaxModuleToEnvConnector
from ray_utilities.learners import mix_learners
from ray_utilities.learners.leaner_with_debug_connector import LearnerWithDebugConnectors
from ray_utilities.setup.algorithm_setup import AlgorithmSetup
from sympol import configs
from sympol.config_types.params_types import SympolCatalogOptions
from sympol.rllib_port.core.jax_learner import JaxPPOLearnerWithLegacy
from sympol.rllib_port.core.sympol_catalog import SympolJaxPPOCatalog
from sympol.rllib_port.core.sympol_module import SympolPPOModule
from sympol.rllib_port.extended_args import SympolArgumentParser

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.core.learner import Learner

    from ray_utilities.setup.experiment_base import NamespaceType

logger = logging.getLogger(__name__)


class SympolSetup(AlgorithmSetup[SympolArgumentParser, PPOConfig, PPO]):
    PROJECT = "SYMPOL"

    GROUP = "NO_GROUP_SET"

    # @property
    # def group_name(self) -> str:
    #    agent_type = self.args.agent_type
    #    if agent_type.lower() == "mlp":
    #        # multiple mlp variants exist
    #        agent_type = f"{agent_type}({self.PROJECT})"
    #    return "_".join([agent_type, self.args.env_type, ("-test" if self.args.test else "")])

    N_STEPS_DEFAULT = 512

    @property
    def model_identifier(self) -> str:
        if self.args.actor == "sympol":
            model_identifier = "-".join([str(self.args.depth), str(self.args.n_estimators), str(self.args.seed)])
        elif self.args.actor != "mlp":
            model_identifier = "-".join([str(self.args.depth), str(self.args.seed)])
        else:
            model_identifier = str(self.args.seed)
        return model_identifier

    def create_parser(self, config_files=None):
        self.parser = SympolArgumentParser()
        return self.parser

    @staticmethod
    @deprecated("legacy")
    def get_minibatch_size(args):
        if False:
            # if we have a trial sample this instead
            return
        if not args.use_best_config:
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
    def _model_config_from_args(cls, args: NamespaceType[SympolArgumentParser]) -> dict[str, Any] | None:
        model_config = super()._model_config_from_args(args) or {}
        # We want all of SympolCatalogOptions in there
        for k in SympolCatalogOptions.__annotations__.keys():
            if hasattr(args, k):
                model_config[k] = getattr(args, k)
            # not supported for 3.10 and typing_extensions 4.15
            elif sys.version_info >= (3, 11) and k in SympolCatalogOptions.__required_keys__:
                raise AttributeError(f"Args has no attribute {k} required for SympolCatalogOptions")
            elif k in SympolCatalogOptions.__required_keys__:
                annots = SympolCatalogOptions.__annotations__
                if k in annots and type_get_origin(annots[k]) is NotRequired:
                    # optional field
                    continue
                raise AttributeError(f"Args has no attribute {k} required for SympolCatalogOptions")
        return model_config

    @classmethod
    def _config_from_args(cls, args, base=None):
        all_data = args.as_dict() if hasattr(args, "as_dict") else vars(args).copy()
        all_data = {k: v for k, v in all_data.items() if not ismethod(v)}
        config, _spec = create_algorithm_config(
            args,
            env_type=args.env_type,
            module_class=SympolPPOModule,
            catalog_class=SympolJaxPPOCatalog,
            # WTF we we store all data in here?
            model_config=cls._model_config_from_args(args) or None,
            framework="torch",  # cannot use "jax" here
            discrete_eval=False,
            base_config=base,
        )
        add_callbacks_to_config(config, cls.get_callbacks_from_args(args))
        # NOTE: Incomplete automcompletion kwargs -> super()
        # Algorithm settings
        # Wether to add DebugConnectors to the pipelines, modify file in place currently.
        DEBUG_CONNECTORS = {"env_to_module": False, "module_to_env": False, "learner": False}
        config.env_runners(
            # env -> module
            add_default_connectors_to_env_to_module_pipeline=False,
            # NOTE: Connectors pin the algorithm in the current state, call after environment(...)
            env_to_module_connector=EnvToModuleWithoutNumpyConnector(config, debug=DEBUG_CONNECTORS["env_to_module"]),
            # module -> env
            add_default_connectors_to_module_to_env_pipeline=False,
            module_to_env_connector=MakeJaxModuleToEnvConnector(
                config,
                key=jax.random.fold_in(jax.random.PRNGKey(args.seed), sum(map(ord, "module_to_env_connector"))),
                debug=DEBUG_CONNECTORS["module_to_env"],
            ),
            # TODO: Should set this in the defaults of the submodule
            num_envs_per_env_runner=4,  # env_context.vector_index
            num_env_runners=2 if args.parallel else 0,  # env_context.worker_index
            num_cpus_per_env_runner=2 if args.parallel else 1,
        )
        # training settings
        learner_mix: list[type[Learner]] = [JaxPPOLearnerWithLegacy]
        if not args.keep_masked_samples:
            from ray_utilities.learners.remove_masked_samples_learner import RemoveMaskedSamplesLearner  # noqa: PLC0415

            learner_mix.insert(0, RemoveMaskedSamplesLearner)
        if DEBUG_CONNECTORS["learner"]:  # NOTE: Must always be the first in the mix
            learner_mix.insert(0, LearnerWithDebugConnectors)
        cast("AlgorithmConfig", config).training(
            add_default_connectors_to_learner_pipeline=True,
            # NOTE: Ray has a wrong typing for learner_connector
            # This is the size the learner receives per _update
            # Legacy minibatches are done in the learner
            learner_config_dict={
                "rng_key": jax.random.fold_in(jax.random.PRNGKey(args.seed), sum(map(ord, "learner"))),
                "accumulate_gradients_every": args.accumulate_gradients_every,
                "legacy": args.legacy,
                "no_numpy_to_tensor_connector": True,  # Disables GeneralAdvantageEstimation._numpy_to_tensor_connector
                "_debug_connectors": DEBUG_CONNECTORS["learner"],  # Not Implemented yet
            },
        )
        try:  # new upcoming interface  # TODO: Check when this is changed ray 2.50+
            config.learners(learner_class=mix_learners(learner_mix))  # pyright: ignore[reportCallIssue]
        except TypeError:  # old interface
            config.training(learner_class=mix_learners(learner_mix))
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

    def _create_trainable(self):
        return super()._create_trainable()

    @classmethod
    def get_algorithm_classes(cls, args):
        if args.algorithm == "dqn":
            raise NotImplementedError("DQN not implemented for SYMPOL yet")
        return PPOConfig, PPO

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
        assert isinstance(suggested_params, dict)
        return cast("dict", suggested_params)


if TYPE_CHECKING:
    SympolSetup(None)
