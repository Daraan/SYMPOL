import sys
from typing import TYPE_CHECKING
from unittest import mock

from rllib_port.core.sympol_module import SympolPPOModule
from rllib_port.sympol_setup import SympolSetup
from tests._test_utils import SetupDefaults, args_train_no_tuner
from utils.envs import build_env

if TYPE_CHECKING:
    from ray.rllib.algorithms.ppo import PPOConfig


class TestTraining(SetupDefaults):
    def setUp(self) -> None:
        super().setUp()
        with mock.patch.object(sys, "argv", ["file.py", "--agent_type", "sympol"]):
            self._SETUP = SympolSetup(init_param_space=False)
            self._SETUP.args.episodes = 20
            self._SETUP.args.render_each_eval = False
            self._SETUP.args.test = True
            self._SETUP.args.comet = False
            self._SETUP.args.wandb = False
            self._ALGORITHM_CONFIG: PPOConfig
            self._ALGORITHM_CONFIG = config = self._SETUP.config  # type: ignore[assignment]
            config.training(
                train_batch_size_per_learner=18, minibatch_size=18, shuffle_batch_per_epoch=False, num_epochs=1
            )
            config.learners(num_gpus_per_learner=0, num_cpus_per_learner=1)
            config.env_runners(num_envs_per_env_runner=3)
            """
            env_to_module_connector: ((EnvType) -> (ConnectorV2 | List[ConnectorV2])) | None = NotProvided,
            module_to_env_connector: ((EnvType, RLModule) -> (ConnectorV2 | List[ConnectorV2])) | None = NotProvided,
            add_default_connectors_to_env_to_module_pipeline: bool | None = NotProvided,
            add_default_connectors_to_module_to_env_pipeline: bool | None = NotProvided,
            """
            self._ENV = build_env("CartPole-v1", 2)
            self._RL_MODULE_SPEC = self._ALGORITHM_CONFIG.get_rl_module_spec(self._ENV)
            self._RL_MODULE = self._RL_MODULE_SPEC.build()
            self.assertEqual(self._RL_MODULE_SPEC.module_class, SympolPPOModule)
            config.learner_config_dict["legacy_minibatch_size"] = 8
            self.failIf(
                config.learner_config_dict.get("legacy_minibatch_size", float("-inf")) > config.minibatch_size,
                "Pretest: Minibatch size is larger than train batch size",
            )

    @args_train_no_tuner
    def test_trainable(self):
        # with self.subTest("No parameters"):
        #    _result = trainable({})
        with self.subTest("With parameters"):
            setup = SympolSetup(init_param_space=True)
            trainable = setup.create_trainable()
            self.assertIsNotNone(setup.args.seed)
            params = setup.sample_params()
            _result = trainable(params)
