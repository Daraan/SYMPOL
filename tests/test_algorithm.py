import sys
from typing import TYPE_CHECKING, cast
from unittest import mock
import unittest

import gymnasium as gym

from rllib_port.sympol.sympol_model import SympolRLModel
from rllib_port.sympol.sympol_module import SympolPPOModule
from rllib_port.sympol.sympol_setup import SympolSetup
from tests._test_utils import SetupDefaults
from utils.envs import build_env

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.algorithms.ppo.ppo import PPOConfig
    from ray.rllib.connectors.env_to_module import EnvToModulePipeline


class AlgorithmTests(SetupDefaults):
    def setUp(self) -> None:
        super().setUp()
        with mock.patch.object(sys, "argv", ["file.py", "--agent_type", "sympol"]):
            self._SETUP = SympolSetup(init_param_space=False)
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

    def test_algorithm_build_units(self):
        with mock.patch.object(sys, "argv", ["file.py", "--agent_type", "sympol"]):
            setup = SympolSetup(init_param_space=False)
            algorithm_config = setup.config
            algorithm_config.framework("torch")
            cast("AlgorithmConfig", algorithm_config).training(
                add_default_connectors_to_learner_pipeline=True,  # maybe false
            )
            """
            env_to_module_connector: ((EnvType) -> (ConnectorV2 | List[ConnectorV2])) | None = NotProvided,
            module_to_env_connector: ((EnvType, RLModule) -> (ConnectorV2 | List[ConnectorV2])) | None = NotProvided,
            add_default_connectors_to_env_to_module_pipeline: bool | None = NotProvided,
            add_default_connectors_to_module_to_env_pipeline: bool | None = NotProvided,
            """
            env = gym.make("CartPole-v1")
            rl_module_spec = algorithm_config.get_rl_module_spec()
            self.assertEqual(rl_module_spec.module_class, SympolPPOModule)
            # Connectors:

            with self.subTest("learner_connector"):
                learner_connector_pipe = algorithm_config.build_learner_connector(
                    self._OBSERVATION_SPACE, self._ACTION_SPACE
                )
            with self.subTest("env_to_module_connector"):
                env_to_module: EnvToModulePipeline = algorithm_config.build_env_to_module_connector(env)
            with self.subTest("module_to_env_connector"):
                module_to_env_connector = algorithm_config.build_module_to_env_connector(env)

            # NOTE: constructs an RLModule
            with self.subTest("learner"):
                learner = algorithm_config.build_learner(env=env)
                learner_group = algorithm_config.build_learner_group(rl_module_spec=rl_module_spec)

    # @unittest.skip("Skip this test for now")
    def test_algorithm_build(self):
        with mock.patch.object(sys, "argv", ["file.py", "--agent_type", "sympol"]):
            setup = SympolSetup(init_param_space=False)
            algorithm_config = setup.config
            algorithm_config.framework("torch")
            algorithm_config.build_algo()

    def test_algorithm_train(self):
        with mock.patch.object(sys, "argv", ["file.py", "--agent_type", "sympol"]):
            setup = SympolSetup(init_param_space=False)
            algorithm_config = setup.config
            algorithm_config.training(train_batch_size_per_learner=32, minibatch_size=8, num_epochs=2)
            algo = algorithm_config.build_algo()
            algo.train()

    def test_evaluate(self):
        result = self._ALGORITHM_CONFIG.build_algo().evaluate()

    def test_step(self):
        result = self._ALGORITHM_CONFIG.build_algo().step()

    def test_module_to_env(self):
        module_to_env = self._ALGORITHM_CONFIG.build_module_to_env_connector(self._ENV)
        model = SympolRLModel(obs_dim=2, action_dim=self._ACTION_DIM, config=self._DEFAULT_CONFIG_DICT)  # pyright: ignore[reportArgumentType]
        actor_state2 = model.init_state(self._ACTOR_KEY, self._ENV_SAMPLE)
        out = model({"state": actor_state2, "obs": self._DEFAULT_INPUT})

        from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
        from ray.rllib.env.single_agent_episode import SingleAgentEpisode
        from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule

        episodes = [SingleAgentEpisode()]
        import numpy as np

        module_to_env(
            rl_module=self._RL_MODULE.as_multi_rl_module(),  # self._RL_MODULE,
            batch={"default_policy": {"action_dist_inputs": out, "actions": np.array([1])}, "actions": np.array([1])},
            episodes=episodes,
            explore=False,
        )

    def test_env_to_module(self):
        env_to_module: EnvToModulePipeline = self._ALGORITHM_CONFIG.build_env_to_module_connector(self._ENV)
        env_to_module(rl_module=self._RL_MODULE, episodes=[], explore=False)

    def test_learner_connector(self):
        learner_connector = self._ALGORITHM_CONFIG.build_learner_connector(self._OBSERVATION_SPACE, self._ACTION_SPACE)
        # no batch
        no_out = learner_connector(rl_module=self._RL_MODULE.as_multi_rl_module(), episodes=[])
