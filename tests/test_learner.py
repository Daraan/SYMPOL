from typing import TYPE_CHECKING, cast

import jax

from ray_utilities.testing_utils import get_leafpath_value
from sympol.rllib_port.sympol_setup import SympolSetup
from tests._test_utils import SympolSetupDefaults, sympol_patch_args

if TYPE_CHECKING:
    from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner

    from sympol.rllib_port.core.sympol_module import SympolPPOModule


class TestSympolLearner(SympolSetupDefaults):
    @sympol_patch_args("--accumulate_gradients_every", "2")
    def test_step_with_gradient_accumulation(self):
        with SympolSetup() as setup:
            # Only one step, to accumulate on every second algo.step call
            setup.config.training(num_epochs=1, train_batch_size_per_learner=128, minibatch_size=128)
            # self.assertEqual(setup.args.accumulate_gradients_every, 2)
            # self.assertEqual(setup.config.learner_config_dict["accumulate_gradients_every"], 2)
        algo = setup.build_algo()
        assert algo.config
        self.assertEqual(algo.config.minibatch_size, 128)
        self.assertEqual(algo.config.num_epochs, 1)
        self.assertEqual(algo.config.train_batch_size_per_learner, 128)
        env_runner = cast("SingleAgentEnvRunner", algo.env_runner_group.local_env_runner)  # type: ignore[attr-defined]
        _runner_module: SympolPPOModule = env_runner.module  # pyright: ignore[reportAssignmentType]
        learner = algo.learner_group._learner  # pyright: ignore[reportOptionalMemberAccess]
        learner_module: SympolPPOModule = learner.module[  # pyright: ignore[reportAssignmentType, reportOptionalMemberAccess]
            "default_policy"
        ]
        states_step0_no_copy = learner_module.states
        states_step0 = learner_module.states.copy()

        for key in ("actor", "critic"):
            with self.subTest(f"Check initial state {key}"):
                self.util_test_state_equivalence(
                    states_step0_no_copy[key],
                    states_step0[key],
                    ignore=[],
                    msg=f"{key}: states_step0 copy vs no copy",
                )

        # Step 1 - states should not change
        algo.step()
        states_step1 = learner_module.states.copy()
        for key in ("actor",):  # Currently critic does not accumulate gradients
            self.util_test_state_equivalence(
                states_step0[key],
                states_step1[key],
                ignore=["step", "grad_accum"],  # grad_accum should change
                ignore_leaves=["count"],
                msg=f"Step 0 vs step 1 - weights should not change (accumulating gradients) for {key}",
            )
            # Check Optimizer step
            for opt_state_leaves in (
                jax.tree.leaves_with_path(states_step0[key].opt_state),
                jax.tree.leaves_with_path(states_step1[key].opt_state),
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
                states_step0[key].grad_accum,
                jax.tree_util.tree_map(jax.numpy.zeros_like, states_step0[key].grad_accum),
                msg=f"Step 0 grad_accum should be zero for {key}",
            )
            # Step 1 should not be zeros anymore
            with self.assertRaisesRegex(AssertionError, "Max relative difference: inf"):
                self.util_test_tree_equivalence(
                    states_step1[key].grad_accum,
                    jax.tree_util.tree_map(jax.numpy.zeros_like, states_step1[key].grad_accum),
                    msg=f"Step 1 grad_accum should not be zeros for {key}",
                )

        # Step 2
        algo.step()
        states_step2 = learner_module.states.copy()
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
