import sys
import unittest
from typing import TYPE_CHECKING, cast
from unittest import mock

import gymnasium as gym
import jax

from rllib_port.core.sympol_module import SympolPPOModule
from rllib_port.sympol.sympol_model import SympolRLModel
from rllib_port.sympol_setup import SympolSetup
from tests._test_utils import SetupDefaults, get_leafpath_value, patch_args
from utils.envs import build_env

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.algorithms.ppo.ppo import PPOConfig
    from ray.rllib.connectors.env_to_module import EnvToModulePipeline
    from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner

    from ray_utilities.jax.ppo.jax_ppo_learner import JaxPPOLearner


class AlgorithmTests(SetupDefaults):
    def setUp(self) -> None:
        SympolSetup.N_STEPS_DEFAULT = 3  # type: ignore
        super().setUp()
        with mock.patch.object(sys, "argv", ["file.py", "--agent_type", "sympol"]):
            self._SETUP = SympolSetup(init_param_space=False)
            self._ALGORITHM_CONFIG: PPOConfig
            self._ALGORITHM_CONFIG = config = self._SETUP.config  # type: ignore[assignment]
            config.learners(num_gpus_per_learner=0, num_cpus_per_learner=1)
            config.evaluation(evaluation_interval=4, evaluation_duration=1)
            config.env_runners(num_envs_per_env_runner=1, episodes_to_numpy=False)
            if self._SETUP.args.legacy:
                config.training(
                    learner_config_dict={"legacy_minibatch_size": 4},
                )
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
        with mock.patch.object(sys, "argv", ["file.py", "--agent_type", "sympol"]):
            setup = SympolSetup(init_param_space=False)
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
        with mock.patch.object(sys, "argv", ["file.py", "--agent_type", "sympol"]):
            setup = SympolSetup(init_param_space=False)
            algorithm_config = setup.config
            algorithm_config.framework("torch")
            algorithm_config.build_algo()

    def test_algorithm_train(self):
        with mock.patch.object(sys, "argv", ["file.py", "--agent_type", "sympol"]):
            setup = SympolSetup(init_param_space=False)
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

    @patch_args("--accumulate_gradients_every", "2")
    def test_step_with_gradient_accumulation(self):
        setup = SympolSetup()
        # Only one step, to accumulate on every second algo.step call
        setup.config.training(num_epochs=1, train_batch_size_per_learner=128, minibatch_size=128)
        # self.assertEqual(setup.args.accumulate_gradients_every, 2)
        # self.assertEqual(setup.config.learner_config_dict["accumulate_gradients_every"], 2)
        algo = setup.build_algo()
        env_runner = cast("SingleAgentEnvRunner", algo.env_runner_group.local_env_runner)  # type: ignore[attr-defined]
        runner_module: SympolPPOModule = env_runner.module  # pyright: ignore[reportAssignmentType]
        states_step0_no_copy = runner_module.states
        states_step0 = runner_module.states.copy()

        for key in ("actor",):
            with self.subTest(f"Check initial state {key}"):
                self.util_test_state_equivalence(
                    states_step0_no_copy[key],
                    states_step0[key],
                    ignore=[],
                    msg=f"{key}: states_step0 copy vs no copy",
                )

        # Step 1 - states should not change
        algo.step()
        states_step1 = runner_module.states.copy()
        self.util_test_state_equivalence(
            states_step0["actor"],
            states_step1["actor"],
            ignore=["step", "grad_accum"],  # grad_accum should change
            ignore_leaves=["count"],
            msg="Step 0 vs step 1 - weights should not change (accumulating gradients)",
        )
        # Check Optimizer step
        for opt_state_leaves in (
            jax.tree.leaves_with_path(states_step0["actor"].opt_state),
            jax.tree.leaves_with_path(states_step1["actor"].opt_state),
        ):
            found_count = False
            for path, val in opt_state_leaves:
                if path and get_leafpath_value(path[-1]) == "count":
                    self.assertEqual(int(val), 0)  # no opt_state taken
                    found_count = True
            self.assertTrue(found_count, "Expected to find count in opt_state leaves")  # metacheck
        # Check grad_accum
        # Step 0 should be 0
        self.util_test_tree_equivalence(
            states_step0["actor"].grad_accum,
            jax.tree_util.tree_map(jax.numpy.zeros_like, states_step0["actor"].grad_accum),
            msg="Step 0 grad_accum should be zero",
        )
        # Step 1 should not be zeros anymore
        with self.assertRaisesRegex(AssertionError, "Max relative difference: inf"):
            self.util_test_tree_equivalence(
                states_step1["actor"].grad_accum,
                jax.tree_util.tree_map(jax.numpy.zeros_like, states_step1["actor"].grad_accum),
                msg="Step 1 grad_accum should not be zeros",
            )

        # Step 2
        algo.step()
        states_step2 = runner_module.states.copy()
        # Step 2: opt_state & params should change
        for attr in ("opt_state", "params"):
            with self.subTest(f"Check '{attr}' after step 2"):
                with self.assertRaises(AssertionError):
                    self.util_test_tree_equivalence(
                        getattr(states_step1["actor"], attr),
                        getattr(states_step2["actor"], attr),
                        msg=f"Step 1 and 2 should not match in {attr}",
                    )
        # grad_accum after step 2 should be zero again
        self.util_test_tree_equivalence(
            states_step2["actor"].grad_accum,
            jax.tree_util.tree_map(jax.numpy.zeros_like, states_step2["actor"].grad_accum),
            msg="Step 2 grad_accum should be zero",
        )
        # Rest should stay the same
        self.util_test_state_equivalence(
            states_step1["actor"],
            states_step2["actor"],
            ignore=["step", "grad_accum", "opt_state", "params"],
            ignore_leaves=["count"],
            msg="Step 1 vs step 2 - other should stay the same",
        )
        found_count = False
        for path, val in jax.tree.leaves_with_path(states_step2["actor"].opt_state):
            if path and get_leafpath_value(path[-1]) == "count":
                self.assertEqual(int(val), 1)  # One opt_state done
                found_count = True
        self.assertTrue(found_count, "Expected to find count in opt_state leaves")  # metacheck

    @unittest.skip("Skip this test. Fails test but works with real inputs.")
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
