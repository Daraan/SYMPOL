"""Unit tests for the TopTrialScheduler."""

from __future__ import annotations

import logging
import time
import unittest
from unittest import mock
from unittest.mock import MagicMock

import pytest
import ray
from ray import tune
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.tune.schedulers.pbt import logger as ray_pbt_logger

from ray_utilities.constants import (
    EPISODE_RETURN_MEAN_EMA,
    EVAL_METRIC_RETURN_MEAN_EMA,
    NUM_ENV_STEPS_PASSED_TO_LEARNER,
    NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME,
)
from ray_utilities.misc import is_pbar, raise_tune_errors
from ray_utilities.runfiles import run_tune
from ray_utilities.testing_utils import (
    DisableLoggers,
    InitRay,
    SetupWithCheck,
    TrainableWithChecks,
)
from ray_utilities.tune.scheduler.grouped_top_pbt_scheduler import GroupedTopPBTTrialScheduler
from ray_utilities.tune.scheduler.top_pbt_scheduler import CyclicMutation, KeepMutation
from sympol._test_utils import SympolTestHelpers, sympol_patch_args
from sympol.rllib_port.core.sympol_pbt_setup import SympolPBTSetup

logger = logging.getLogger(__name__)

patch_args = sympol_patch_args


class TestGroupedTopPBTIntegration(InitRay, SympolTestHelpers, DisableLoggers):
    """Integration test for GroupedTopPBTTrialScheduler using run_tune.

    Note: This test requires Ray to be initialized. Run with InitRay if needed.
    """

    @pytest.mark.length(speed="medium")
    @mock.patch("wandb.Api", new=MagicMock())
    @mock.patch("ray_utilities.callbacks.wandb.wandb_api", new=MagicMock())
    @pytest.mark.timeout(320)
    def test_run_tune_with_grouped_top_pbt_scheduler(self):
        """Test GroupedTopPBTTrialScheduler with run_tune using grouped trials."""
        # Need to import here to avoid circular imports

        # Skip if Ray not initialized
        if not ray.is_initialized():
            pytest.skip("Ray not initialized")

        original_exploit = GroupedTopPBTTrialScheduler._exploit
        perturbation_interval = 100
        best_group_idx = 1  # Group 1 (with lr=0.01) will be best

        # Create 3 learning rates, each with 2 seeds = 6 trials total
        # 3 groups of 2 trials each
        learning_rates = (0.001, 0.01, 0.005)  # Group 1 (lr=0.01) will have highest scores
        num_seeds = 2

        num_exploits = 0
        group_exploit_counts = dict.fromkeys(learning_rates, 0)

        # Fake results: scores depend on learning rate and step
        # Group with lr=0.01 will consistently perform best
        fake_results: dict[float, dict[int, float]] = {
            learning_rates[0]: {  # lr=0.001: scores 1, 2, 3, ...
                v: v // perturbation_interval for v in range(perturbation_interval, 401, perturbation_interval)
            },
            learning_rates[1]: {  # lr=0.01: scores 21, 22, 23, ... (BEST)
                v: v // perturbation_interval + 20 for v in range(perturbation_interval, 401, perturbation_interval)
            },
            learning_rates[2]: {  # lr=0.005: scores 6, 7, 8, ...
                v: v // perturbation_interval + 5 for v in range(perturbation_interval, 401, perturbation_interval)
            },
        }

        race_conditions = 0

        def test_exploit_function(self: GroupedTopPBTTrialScheduler, tune_controller, trial, trial_to_clone) -> None:
            """Verify group-based exploitation logic."""
            nonlocal num_exploits, race_conditions
            num_exploits += 1

            trial_lr = trial.config.get("lr", trial.config.get("cli_args", {}).get("lr"))
            clone_lr = trial_to_clone.config.get("lr", trial_to_clone.config.get("cli_args", {}).get("lr"))

            logger.info(
                "Exploit #%d: trial lr=%s → clone lr=%s at step %s",
                num_exploits,
                trial_lr,
                clone_lr,
                self._trial_state[trial].last_train_time,
            )

            if self._trial_state[trial].last_perturbation_time % perturbation_interval != 0:
                race_conditions += 1
                logger.warning(
                    "Exploit at step %s not at perturbation interval (race condition)",
                    self._trial_state[trial].last_perturbation_time,
                )
            else:
                # Trial being exploited should NOT be from best group
                assert trial_lr != learning_rates[best_group_idx], (
                    f"Trial with lr={trial_lr} should not be in best group"
                )
                # Verify that lower-performing trial exploits higher-performing group
                # The best group (lr=0.01) should be cloned
                assert clone_lr == learning_rates[best_group_idx], (
                    f"Expected clone from best group lr={learning_rates[best_group_idx]}, got lr={clone_lr}"
                )

                group_exploit_counts[trial_lr] += 1

            # Call original exploit function
            original_exploit(self, tune_controller, trial, trial_to_clone)

        GroupedTopPBTTrialScheduler._exploit = test_exploit_function

        class CheckTrainableForGroupedPBT(TrainableWithChecks):
            """Custom trainable that returns predetermined scores based on learning rate."""

            debug_step = False
            use_pbar = False

            def step(self):  # pyright: ignore[reportIncompatibleMethodOverride]
                """Return fake results based on learning rate."""
                # Get learning rate from config
                lr = self.algorithm_config.lr or self.config.get("cli_args", {}).get("lr", 0.001)
                print("DEBUG: Got lr:", lr)

                self._current_step += self.algorithm_config.train_batch_size_per_learner
                result = {ENV_RUNNER_RESULTS: {}, EVALUATION_RESULTS: {ENV_RUNNER_RESULTS: {}}}
                result[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_PASSED_TO_LEARNER_LIFETIME] = self._current_step
                result[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_PASSED_TO_LEARNER] = (
                    self.algorithm_config.train_batch_size_per_learner
                )
                result[ENV_RUNNER_RESULTS][NUM_ENV_STEPS_SAMPLED_LIFETIME] = self._current_step + 2

                # from ray_utilities.testing_utils import remote_breakpoint
                # remote_breakpoint()
                # Return score from fake_results
                result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN] = fake_results[lr][
                    self._current_step
                ]
                result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN_EMA] = result[EVALUATION_RESULTS][
                    ENV_RUNNER_RESULTS
                ][EPISODE_RETURN_MEAN]
                result["_checking_class_"] = "CheckTrainableForGroupedPBT"

                logger.info(
                    "LR: %s, step %s, result: %s",
                    lr,
                    self._current_step,
                    result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN],
                )
                result["current_step"] = self._current_step

                if is_pbar(self._pbar):
                    self._pbar.update(1)
                    self._pbar.set_description(
                        f"Step: {self._current_step} lr={lr} "
                        f"result={result[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]}"
                    )

                time.sleep(2)  # Simulate some work
                return result

        ray_pbt_logger.setLevel(logging.DEBUG)

        with patch_args(
            # Main experiment args
            "--tune", "lr",
            # Meta arguments
            "--num_samples", num_seeds,  # 2 seeds per learning rate
            "--num_jobs", len(learning_rates) * num_seeds,  # 6 total trials
            "--batch_size", perturbation_interval,
            "--minibatch_size",
            perturbation_interval,
            "--total_steps", perturbation_interval * 3,
            "--use_exact_total_steps",
            # TODO: We want all groups to have the same seed sequence
            "--env_seeding_strategy", "sequential",  # Different seeds per trial
            # Constant
            "--seed", "42",
            "--log_level", "DEBUG",
            "--log_stats", "most",
            "--no_dynamic_eval_interval",
            "--test",
            "--num_envs_per_env_runner", 1,
            "pbt",
            "--quantile_fraction", "0.34",  # Top 1/3 of groups (1 out of 3)
            "--perturbation_interval", perturbation_interval,
        ):  # fmt: skip
            Setup = SetupWithCheck(CheckTrainableForGroupedPBT, SympolPBTSetup)
            with Setup(config_files=[]) as setup:
                setup.config.training(num_epochs=1)
                setup.config.reporting(min_sample_timesteps_per_iteration=10)
            assert setup.args.mode == "max"
            assert setup.args.metric == EVAL_METRIC_RETURN_MEAN_EMA
            assert setup.args.command.mode == "max"
            assert setup.args.command.metric == EVAL_METRIC_RETURN_MEAN_EMA
            self.assertEqual(setup.args.command.quantile_fraction, 0.34)

            # Use grid search for learning rate
            setup.param_space["lr"] = tune.grid_search(learning_rates)

            # Set mutations
            assert setup.args.command
            setup.args.command.set_hyperparam_mutations(
                {
                    "lr": CyclicMutation(learning_rates),
                    "depth": KeepMutation(2),
                }
            )

            # Create custom scheduler
            setup.args.command.to_scheduler = lambda *args, **kwargs: GroupedTopPBTTrialScheduler(
                # metric="episode_reward_mean",
                # mode="max",
                perturbation_interval=perturbation_interval,
                quantile_fraction=0.34,  # Top 1 out of 3 groups
                hyperparam_mutations=setup.args.command.hyperparam_mutations,
                num_samples=num_seeds,  # Expected trials per group
                synch=True,
            )

            results = run_tune(setup)
            raise_tune_errors(results)

            # Verify all results are from our custom trainable
            self.assertTrue(
                all(result.metrics["_checking_class_"] == "CheckTrainableForGroupedPBT" for result in results)
            )

            # Expected exploitations:
            # - 3 steps total (at 100, 200, 300)
            # - At each perturbation interval, lower quantile groups exploit upper quantile
            # - With 3 groups and quantile_fraction=0.34, we have 1 upper group (best) and 2 lower groups
            # - Each lower group has 2 trials, so 4 total exploitations per interval
            # - But at step 300, training ends, so only 2 intervals have full exploitation
            expected_exploits = 2 * 2 * num_seeds  # 2 lower groups x 2 trials x 2 intervals

            logger.info("Total exploits: %d, Expected: %d", num_exploits, expected_exploits)
            self.assertGreaterEqual(num_exploits, expected_exploits - 2)  # Allow some race conditions
            self.assertLessEqual(num_exploits, expected_exploits + 2)

            # Check that at most a few race conditions happened
            self.assertLessEqual(race_conditions, 2)

            # Verify all final configs have depth=2
            self.assertTrue(all(r.config["depth"] == 2 for r in results))

            logger.info("Group exploit counts: %s", group_exploit_counts)

        # Restore original exploit function
        GroupedTopPBTTrialScheduler._exploit = original_exploit


if __name__ == "__main__":
    unittest.main()
