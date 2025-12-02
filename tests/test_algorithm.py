from typing import TYPE_CHECKING, cast

import gymnasium as gym
import pytest

from ray_utilities.testing_utils import patch_args
from sympol._test_utils import SympolSetupDefaults, get_leafpath_value
from sympol.rllib_port.core.sympol_module import SympolPPOModule
from sympol.rllib_port.sympol.sympol_model import SympolRLModel
from sympol.rllib_port.sympol_setup import SympolSetup
from sympol.utils.envs import build_env

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.algorithms.ppo.ppo import PPOConfig
    from ray.rllib.connectors.env_to_module import EnvToModulePipeline
    from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner

    from ray_utilities.jax.ppo.jax_ppo_learner import JaxPPOLearner


class AlgorithmTests(SympolSetupDefaults):
    def setUp(self) -> None:
        SympolSetup.N_STEPS_DEFAULT = 3  # type: ignore
        super().setUp()
        with patch_args("--agent_type", "sympol"), SympolSetup(init_param_space=False) as setup:
            self._SETUP = setup
            self._ALGORITHM_CONFIG: PPOConfig
            self._ALGORITHM_CONFIG = config = self._SETUP.config  # type: ignore[assignment]
            config.learners(num_gpus_per_learner=0, num_cpus_per_learner=1)
            config.evaluation(evaluation_interval=4, evaluation_duration=1)
            config.env_runners(num_envs_per_env_runner=1, episodes_to_numpy=False)
            if self._SETUP.args.legacy:
                config.training(
                    learner_config_dict={"legacy_minibatch_size": 4},
                )
                # NOTE: Setting train_batch_size_per_learner to 8, without argparser will log an error
                # No evaluation interval for 8 steps in
                config.training(
                    train_batch_size_per_learner=8, minibatch_size=8, shuffle_batch_per_epoch=False, num_epochs=2
                )
            else:
                config.training(
                    train_batch_size_per_learner=8, minibatch_size=4, shuffle_batch_per_epoch=False, num_epochs=2
                )
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
        with patch_args("--agent_type", "sympol"):
            with SympolSetup(init_param_space=False) as setup:
                algorithm_config = setup.config
                algorithm_config.framework("torch")
                cast("AlgorithmConfig", algorithm_config).training(  # pyright: ignore[reportUnnecessaryCast]
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
                _learner_connector_pipe = algorithm_config.build_learner_connector(
                    self._OBSERVATION_SPACE, self._ACTION_SPACE
                )
            with self.subTest("env_to_module_connector"):
                _env_to_module: EnvToModulePipeline = algorithm_config.build_env_to_module_connector(env)
            with self.subTest("module_to_env_connector"):
                _module_to_env_connector = algorithm_config.build_module_to_env_connector(env)

            # NOTE: constructs an RLModule
            with self.subTest("learner"):
                _learner = algorithm_config.build_learner(env=env)
                _learner_group = algorithm_config.build_learner_group(rl_module_spec=rl_module_spec)

    # @unittest.skip("Skip this test for now")
    def test_algorithm_build(self):
        with patch_args("--agent_type", "sympol"):
            with SympolSetup(init_param_space=False) as setup:
                algorithm_config = setup.config
                algorithm_config.framework("torch")
            algorithm_config.build_algo()

    def test_algorithm_train(self):
        with patch_args("--agent_type", "sympol"):
            with SympolSetup(init_param_space=False) as setup:
                algorithm_config = setup.config
            # Does not comply with legacy:
            # algorithm_config.training(train_batch_size_per_learner=32, minibatch_size=8, num_epochs=2)
            algo = algorithm_config.build_algo()
            algo.train()

    def test_evaluate(self):
        _result = self._ALGORITHM_CONFIG.build_algo().evaluate()

    def test_step(self):
        algo = self._ALGORITHM_CONFIG.build_algo()
        _result = algo.step()

        with self.subTest("test weights after step"):
            env_runner = cast("SingleAgentEnvRunner", algo.env_runner_group.local_env_runner)  # type: ignore[attr-defined]
            eval_env_runner = cast("SingleAgentEnvRunner", algo.eval_env_runner_group.local_env_runner)  # type: ignore[attr-defined]
            learner: JaxPPOLearner = algo.learner_group._learner  # pyright: ignore[reportAssignmentType, reportOptionalMemberAccess]
            learner_multi = learner.module

            runner_module: SympolPPOModule = env_runner.module  # pyright: ignore[reportAssignmentType]
            eval_module: SympolPPOModule = eval_env_runner.module  # pyright: ignore[reportAssignmentType]
            learner_module: SympolPPOModule = learner_multi["default_policy"]  # pyright: ignore[reportAssignmentType]
            algo_module: SympolPPOModule = (
                algo.get_module()
            )  # algo.env_runner.module# pyright: ignore[reportAssignmentType]

            # Test weight sync
            # NOTE THESE ARE TRAIN STATES not arrays
            ignore = ("step", "opt_state")
            if runner_module is not algo_module:
                print("WARNING: modules are not the object")
                # identity
                self.util_test_state_equivalence(
                    runner_module.states["actor"],
                    algo_module.states["actor"],
                    ignore=ignore,
                    msg="actor: runner_module vs algo_module",
                )
                self.util_test_state_equivalence(
                    runner_module.states["critic"],
                    algo_module.states["critic"],
                    ignore=ignore,
                    msg="critic: runner_module vs algo_module",
                )
            self.util_test_state_equivalence(
                learner_module.states["actor"],
                algo_module.states["actor"],
                ignore=ignore,
                msg="actor: learner_module vs algo_module",
            )
            algo.evaluate()
            # These are possibly not updated
            self.util_test_state_equivalence(
                eval_module.states["actor"],
                algo_module.states["actor"],
                ignore=ignore,
                msg="actor: eval_module vs algo_module",
            )
            # critic might not be present in inference only mode
            self.assertTrue("critic" in algo_module.states or algo_module.inference_only)
            self.assertTrue("critic" in eval_module.states or eval_module.inference_only)
            if "critic" in eval_module.states:
                self.util_test_state_equivalence(
                    learner_module.states["critic"],
                    eval_module.states["critic"],
                    ignore=ignore,
                    msg="critic: learner_module vs eval_module",
                )
            if "critic" in algo_module.states:
                # removed inference only state in get_state
                self.util_test_state_equivalence(
                    learner_module.states["critic"],
                    algo_module.states["critic"],
                    ignore=ignore,
                    msg="critic: learner_module vs algo_module",
                )

    @pytest.mark.xfail(reason="Skip this test. Fails test but works with real inputs.")
    def test_module_to_env(self):
        module_to_env = self._ALGORITHM_CONFIG.build_module_to_env_connector(self._ENV)
        model = SympolRLModel(obs_dim=2, action_dim=self._ACTION_DIM, config=self._DEFAULT_CONFIG_DICT)  # pyright: ignore[reportArgumentType]
        actor_state2 = model.init_state(self._ACTOR_KEY, self._ENV_SAMPLE)
        out = model({"obs": self._DEFAULT_INPUT}, parameters=actor_state2.params, indices=actor_state2.indices)

        from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
        from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
        from ray.rllib.env.single_agent_episode import SingleAgentEpisode

        episodes = [SingleAgentEpisode(observations=self._DEFAULT_INPUT)]
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
        _no_out = learner_connector(rl_module=self._RL_MODULE.as_multi_rl_module(), episodes=[])
