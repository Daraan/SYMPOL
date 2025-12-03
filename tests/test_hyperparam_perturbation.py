from __future__ import annotations

import dataclasses
import unittest
from typing import TYPE_CHECKING, Any, cast

import optax
import pytest

from ray_utilities.jax.jax_learner import JaxLearner
from ray_utilities.testing_utils import InitRay, mock_trainable_algorithm, no_parallel_envs
from sympol._test_utils import SympolTestHelpers
from sympol._test_utils import sympol_patch_args as patch_args
from sympol.rllib_port.core.jax_learner import JaxPPOLearnerWithLegacy
from sympol.rllib_port.sympol_setup import SympolSetup

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import AlgorithmConfig
    from ray.rllib.core.rl_module.rl_module import RLModuleConfig

    from ray_utilities.jax.ppo.compute_ppo_loss import _PPOSettings
    from ray_utilities.jax.ppo.jax_ppo_learner import JaxPPOLearner
    from ray_utilities.training.default_class import TrainableBase
    from sympol.rllib_port.core.algorithms import SympolPPOConfig
    from sympol.rllib_port.core.sympol_module import SympolPPOModule


class TestHyperparamPerturbation(InitRay, SympolTestHelpers, num_cpus=4):
    def get_modules(self, trainable: TrainableBase):
        runner_module = cast("SympolPPOModule", trainable.algorithm.get_module())
        learner_module = cast("SympolPPOModule", trainable.algorithm.learner_group._learner.module["default_policy"])
        return runner_module, learner_module

    def _test_learning_rate_set(self, trainable, target_lr: float, msg: str = ""):
        runner_module, learner_module = self.get_modules(trainable)

        # Check learner module
        learner_actor_state = learner_module.states["actor"]
        lrs_learner = JaxLearner._find_lrs(learner_actor_state.opt_state)
        self.assertTrue(len(lrs_learner) > 0, f"{msg} No hyperparameters found in learner opt_state")
        self.assertTrue(
            any(abs(lr - target_lr) < 1e-6 for lr in lrs_learner),
            f"{msg} Target LR {target_lr} not found in learner opt_state. Found: {optax.tree_utils.tree_get_all_with_path(learner_actor_state.opt_state, 'learning_rate')}",
        )

        # Check runner module
        runner_actor_state = runner_module.states["actor"]
        lrs_runner = JaxLearner._find_lrs(runner_actor_state.opt_state)
        self.assertTrue(len(lrs_runner) > 0, f"{msg} No hyperparameters found in runner opt_state")
        self.assertTrue(
            any(abs(lr - target_lr) < 1e-6 for lr in lrs_runner),
            f"{msg} Target LR {target_lr} not found in runner opt_state. Found: {optax.tree_utils.tree_get_all_with_path(runner_actor_state.opt_state, 'learning_rate')}",
        )

    def _test_grad_clip_set(self, trainable, target_clip: float, msg: str = ""):
        runner_module, learner_module = self.get_modules(trainable)

        for module_name, module in [("learner", learner_module), ("runner", runner_module)]:
            # module.pi is the actor (SympolRLModel)
            actor = module.pi
            config = actor.config

            # Handle potential key differences or defaults
            actual_clip = config.get("grad_clip")
            if actual_clip is None:
                raise KeyError("grad_clip not found, and max_grad_norm handling not implemented")

            self.assertIsNotNone(actual_clip, f"{msg} {module_name}: grad_clip not found in actor config")
            self.assertAlmostEqual(
                actual_clip,
                target_clip,
                places=6,
                msg=f"{msg} {module_name}: Expected grad_clip {target_clip}, got {actual_clip}",
            )

    @no_parallel_envs
    def test_learner_can_set_lr(self):
        target_lr = 0.123

        with patch_args("--lr", "0.1"):
            trainable = SympolSetup().trainable_class()
            learner = cast("JaxLearner", trainable.algorithm.learner_group._learner)
        learner._set_optimizer_lr(None, target_lr)
        opt_states = learner._get_optimizer_state()
        for _, lr in optax.tree_utils.tree_get_all_with_path(opt_states, "learning_rate"):
            self.assertAlmostEqual(
                lr,
                target_lr,
                places=6,
                msg=f"Learner optimizer state learning rate not set correctly. Expected {target_lr}, got {lr}",
            )
        self._test_learning_rate_set(trainable, target_lr)

    @no_parallel_envs
    def test_lr_setting(self):
        # 'Learning rate'
        target_lr = 0.123

        # test cli
        with SympolSetup() as setup:
            basic_config: "AlgorithmConfig" = setup.config
            # Set the learning rate in the config
            ppo_config = setup.config  # for type checking
            basic_config.training(lr=target_lr)
        trainable = setup.trainable_class()
        self._test_learning_rate_set(trainable, target_lr, "AlgorithmConfig lr set")
        trainable.stop()
        del trainable

        # test config override
        trainable2 = setup.trainable_class({"lr": 0.045})
        self._test_learning_rate_set(trainable2, 0.045, "trainable.config lr override")
        with self.assertRaises(AssertionError):
            self._test_learning_rate_set(trainable2, 0.123)
        trainable2.stop()
        del trainable2

        with patch_args("--lr", 0.144):
            setup = SympolSetup()
        trainable3 = setup.trainable_class()
        self._test_learning_rate_set(trainable3, 0.144, "CLI lr override")
        trainable3.stop()
        del trainable3

    @no_parallel_envs
    def test_grad_clip_setting(self):
        target_clip = 0.111
        with SympolSetup() as setup:
            basic_config: "AlgorithmConfig" = setup.config
            basic_config.rl_module(model_config=setup.config._model_config | {"grad_clip": target_clip})
        trainable = setup.trainable_class()
        # self.assertEqual(setup.config.grad_clip, target_clip)
        self.assertEqual(setup.config.model_config["grad_clip"], target_clip)
        self._test_grad_clip_set(trainable, target_clip, "AlgorithmConfig grad_clip set")
        trainable.stop()
        del trainable

        # test config override
        setup = SympolSetup()
        trainable2 = setup.trainable_class({"grad_clip": 0.045})
        self._test_grad_clip_set(trainable2, 0.045, "trainable.config grad_clip override")
        with self.assertRaises(AssertionError):
            self._test_grad_clip_set(trainable2, 0.5)
        trainable2.stop()
        del trainable2

        with patch_args("--grad_clip", 0.144):
            setup = SympolSetup()
        self.assertEqual(setup.args.grad_clip, 0.144)
        self.assertEqual(setup.config.grad_clip, 0.144)
        trainable3 = setup.trainable_class()
        self._test_grad_clip_set(trainable3, 0.144, "CLI grad_clip override")
        del trainable3

        with SympolSetup() as setup:
            basic_config: "AlgorithmConfig" = setup.config
            basic_config.training(grad_clip=target_clip)

        trainable = setup.trainable_class()
        self.assertEqual(setup.config.grad_clip, target_clip)
        self.assertEqual(setup.config.model_config["grad_clip"], target_clip)
        self._test_grad_clip_set(trainable, target_clip, "AlgorithmConfig grad_clip set")
        trainable.stop()

    def _check_module_config_setting(self, trainable, key: str, expected: bool | Any = True):
        runner_module, learner_module = self.get_modules(trainable)

        for module_name, module in [("learner", learner_module), ("runner", runner_module)]:
            # module.pi is the actor (SympolRLModel)
            actor = module.pi
            config = actor.config
            # Cannot check actor_states directly as jit compiled

            self.assertIs(
                module.model_config.get(key),
                expected,
                f"{module_name}: Expected adamW to be {expected}, got {module.model_config.get('adamW')}",
            )

            actual_adamW = config.get(key)
            self.assertIsNotNone(actual_adamW, f"{module_name}: adamW not found in actor config")
            self.assertIs(
                actual_adamW,
                expected,
                f"{module_name}: Expected adamW to be {expected}, got {actual_adamW}",
            )

            if hasattr(module, "vf"):
                critic = module.vf
                critic_config = critic.config
                actual_adamW_critic = critic_config.get(key)
                self.assertIsNotNone(actual_adamW_critic, f"{module_name}: adamW not found in critic config")
                self.assertIs(
                    actual_adamW_critic,
                    expected,
                    f"{module_name}: Expected adamW to be {expected} in critic, got {actual_adamW_critic}",
                )
            else:
                assert module.inference_only

    @pytest.mark.xfail(reason="No priority")
    @no_parallel_envs
    def test_adamW_setting(self):  # noqa: N802
        for on_off in [True, False]:
            with SympolSetup() as setup:
                basic_config: "AlgorithmConfig" = setup.config
                basic_config.rl_module(model_config=setup.config._model_config | {"adamW": on_off})
            trainable = setup.trainable_class()
            self.assertEqual(trainable.algorithm_config.model_config["adamW"], on_off)
            self._check_module_config_setting(trainable, "adamW", on_off)
            trainable.stop()
            del trainable

            # test config override
            setup = SympolSetup()
            trainable2 = setup.trainable_class({"adamW": on_off})
            self.assertEqual(trainable2.algorithm_config.model_config["adamW"], on_off)
            self._check_module_config_setting(trainable2, "adamW", on_off)
            with self.assertRaises(AssertionError):
                self._check_module_config_setting(trainable2, "adamW", not on_off)
            trainable2.stop()
            del trainable2

            with patch_args("--adamW" if on_off else "--no-adamW"):
                setup = SympolSetup()
            trainable3 = setup.trainable_class()
            self.assertEqual(trainable3.algorithm_config.model_config["adamW"], on_off)
            self._check_module_config_setting(trainable3, "adamW", on_off)
            del trainable3

    @pytest.mark.xfail(reason="No priority")
    @no_parallel_envs
    def test_SWA_setting(self):  # noqa: N802
        for on_off in [True, False]:
            with SympolSetup() as setup:
                basic_config: "AlgorithmConfig" = setup.config
                basic_config.rl_module(model_config=setup.config._model_config | {"SWA": on_off})
            trainable = setup.trainable_class()
            self.assertEqual(trainable.algorithm_config.model_config["SWA"], on_off)
            self._check_module_config_setting(trainable, "SWA", on_off)
            trainable.stop()
            del trainable

            # test config override
            setup = SympolSetup()
            trainable2 = setup.trainable_class({"SWA": on_off})
            self.assertEqual(trainable2.algorithm_config.model_config["SWA"], on_off)
            self._check_module_config_setting(trainable2, "SWA", on_off)
            with self.assertRaises(AssertionError):
                self._check_module_config_setting(trainable2, "SWA", not on_off)
            trainable2.stop()
            del trainable2

            with patch_args("--SWA") if on_off else patch_args():
                setup = SympolSetup()
            trainable3 = setup.trainable_class()
            self.assertEqual(trainable3.algorithm_config.model_config["SWA"], on_off)
            self._check_module_config_setting(trainable3, "SWA", on_off)
            del trainable3

    def _compare_config_attributes(
        self,
        config1: SympolPPOConfig | _PPOSettings,
        expected_valued: dict[str, Any],
        msg: str = "",
    ):
        for k, v in expected_valued.items():
            actual_value = getattr(config1, k)
            self.assertEqual(
                actual_value,
                v,
                f"{msg} PPO Config key '{k}' expected value {v}, got {actual_value}",
            )

    # Test further PPO parameters
    def test_ppo_loss_parameters(self):
        ppo_settings1 = {"clip_param": 0.211, "use_kl_loss": True, "vf_loss_coeff": 2.0, "vf_clip_param": 0.123}
        with SympolSetup() as setup:
            setup.config.training(**ppo_settings1)

        self._compare_config_attributes(
            setup.config,
            ppo_settings1,
            msg="Setup config not set correctly",
        )
        trainable = setup.trainable_class()

        #  For CI we can access self._compute_loss_for_modules[module_id].__config for comparison
        learner = cast("JaxPPOLearner", trainable.algorithm.learner_group._learner)
        for module_id in learner.module.keys():
            compute_loss_fn = learner._compute_loss_for_modules[module_id]
            config_during_compile_config: _PPOSettings = getattr(compute_loss_fn, "__config")
            self.assertIsNot(config_during_compile_config, trainable.algorithm_config)
            loss_ppo_settings1 = dataclasses.asdict(config_during_compile_config)
            algo_dict = trainable.algorithm_config.to_dict()
            self.util_test_tree_equivalence(
                algo_dict,
                algo_dict | loss_ppo_settings1,
            )
            self._compare_config_attributes(
                config_during_compile_config,
                ppo_settings1,
                msg=f"Learner compute_loss config for module {module_id} not set correctly",
            )
            self._compare_config_attributes(
                trainable.algorithm_config,
                ppo_settings1,
                msg=f"Setup config for module {module_id} not set correctly",
            )
        trainable.stop()
        del trainable
        del learner

        # Test via input

        ppo_settings2 = {
            "clip_param": 0.333,
            "use_kl_loss": False,
            "vf_loss_coeff": 0.12,
        }
        trainable2 = setup.trainable_class(ppo_settings2)
        ppo_settings2_merged = ppo_settings1 | ppo_settings2
        self._compare_config_attributes(
            trainable2.algorithm_config,
            ppo_settings2_merged,
            msg="trainable.config not set correctly",
        )
        #  For CI we can access self._compute_loss_for_modules[module_id].__config for comparison
        learner = cast("JaxPPOLearner", trainable2.algorithm.learner_group._learner)
        for module_id in learner.module.keys():
            compute_loss_fn = learner._compute_loss_for_modules[module_id]
            config_during_compile_config2: _PPOSettings = getattr(compute_loss_fn, "__config")
            self.assertIsNot(config_during_compile_config2, trainable2.algorithm_config)
            loss_ppo_settings2 = dataclasses.asdict(config_during_compile_config2)
            algo_dict = trainable2.algorithm_config.to_dict()
            self.util_test_tree_equivalence(
                algo_dict,
                algo_dict | loss_ppo_settings2,
            )
            self.assertNotAlmostEqual(config_during_compile_config.clip_param, config_during_compile_config2.clip_param)  # pyright: ignore
            self.assertNotAlmostEqual(
                config_during_compile_config.vf_loss_coeff,  # pyright: ignore[reportPossiblyUnboundVariable]
                config_during_compile_config2.vf_loss_coeff,
            )  # pyright: ignore
            with self.assertRaises(AssertionError):
                self.util_test_tree_equivalence(
                    loss_ppo_settings2,
                    loss_ppo_settings1,  # pyright: ignore[reportPossiblyUnboundVariable]
                )
            self._compare_config_attributes(
                config_during_compile_config2,
                ppo_settings2_merged,
                msg=f"Learner compute_loss config for module {module_id} not set correctly",
            )
            self._compare_config_attributes(
                trainable2.algorithm_config,
                ppo_settings2_merged,
                msg=f"Setup config for module {module_id} not set correctly",
            )
