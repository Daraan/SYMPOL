import sys
from unittest import mock

from rllib_port.sympol.sympol_module import SympolPPOModule
from rllib_port.sympol.sympol_setup import SympolSetup
from tests._test_utils import SetupDefaults
from utils.envs import build_env


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
                train_batch_size_per_learner=18, minibatch_size=6, shuffle_batch_per_epoch=False, num_epochs=1
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

    def test_trainable(self):
        trainable = self._SETUP.create_trainable()
        trainable({})
