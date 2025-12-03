import sys
import unittest
import unittest.mock
from dataclasses import asdict
from inspect import ismethod
from typing import Any

from frozendict import frozendict
from tap import Tap

from ray_utilities.callbacks.algorithm.dynamic_batch_size import DynamicGradientAccumulation
from ray_utilities.callbacks.algorithm.dynamic_buffer_callback import DynamicBufferUpdate
from ray_utilities.callbacks.algorithm.exact_sampling_callback import exact_sampling_callback
from ray_utilities.connectors.remove_masked_samples_connector import RemoveMaskedSamplesConnector
from ray_utilities.learners.remove_masked_samples_learner import RemoveMaskedSamplesLearner
from sympol._test_utils import (
    SympolSetupDefaults,
    args_train_no_tuner,
    clean_args,
    get_required_keys,
    sympol_patch_args,
)
from sympol.args import get_args, get_args_old  # noqa: F401  # avoid circular imports
from sympol.config_types.args_types import SympolCLIArgs
from sympol.config_types.params_types import CLIArgsDict, MLPParams, SDTParams, SympolParams
from sympol.rllib_port.extended_args import SympolArgumentParser
from sympol.rllib_port.sympol_setup import SympolSetup
from tests._original_args import get_original_args

_default_args = SympolCLIArgs()
# NOTE: In vars no_ attributes are removed; with asdict not!
_default_args_dict = frozendict(vars(_default_args))
_default_args_key_set = set(_default_args_dict.keys())


class TestArgObjects(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.maxDiff = 2000

    def test_equivalence_cli_args(self):
        self.maxDiff = 2000
        self.assertEqual(hash(_default_args), hash(SympolCLIArgs()))

        args_same = SympolCLIArgs()
        args_same_dict = asdict(args_same)

        with self.assertRaises(AssertionError):
            # no_ attributes are not present when using vars; with asdict they are
            self.assertEqual(args_same_dict, _default_args_dict)

        same_dict_positive = {k: v for k, v in args_same_dict.items() if not k.startswith("no_")}
        self.assertDictEqual(dict(_default_args_dict), same_dict_positive)
        self.assertEqual(hash(tuple(_default_args_dict.items())), hash(tuple(same_dict_positive.items())))

        # different hash

        args_different = SympolCLIArgs()
        args_different.depth = 2
        args_different_dict = vars(args_different)
        self.assertNotEqual(args_different, _default_args)
        self.assertNotEqual(hash(args_different), hash(_default_args))
        self.assertNotEqual(args_different_dict, _default_args_dict)

        # difference in non-hashable attribute
        args_different_but_same = SympolCLIArgs()
        args_different_but_same.random_trials = _default_args.random_trials + 342
        self.assertEqual(args_different_but_same, _default_args)
        self.assertEqual(hash(args_different_but_same), hash(_default_args))

    @clean_args
    def test_equivalence_argument_parser(self):
        args1 = SympolSetup(init_param_space=False).args
        args2 = SympolSetup(init_param_space=False).args
        self.assertEqual(args1, args2)
        self.assertEqual(hash(args1), hash(args2))

        args_different = SympolSetup(init_param_space=False).args
        args_different.depth = 2
        self.assertNotEqual(args_different, args1)
        self.assertNotEqual(hash(args_different), hash(args1))

        # This seems possible as it inherits from CLIArgs
        args_different_but_same = SympolSetup(init_param_space=False).args
        args_different_but_same.random_trials = args1.random_trials + 342
        self.assertEqual(args_different_but_same, args1)
        self.assertEqual(hash(args_different_but_same), hash(args1))


class TestArgContents(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    @clean_args
    def test_equivalence_tap(self):
        """
        NOTE: This test is runtime dependent on the order of execution of the attributes

        As it is NOT deterministic this test could pass or fail.
        e.g. if adamW; no_adamW is False
        """
        self.maxDiff = None
        parser = SympolArgumentParser()
        args = parser.parse_args()
        # SympolCLIArgs subset of SympolArgumentParser
        default_args = SympolCLIArgs()
        # no - attributes
        self.assertFalse(default_args.adamW)
        self.assertFalse(args.adamW)
        # NOTE: render_env default differs between CLIArgs (True) and ArgumentParser (False)
        self.assertIn(args.render_env, [True, False])
        self.assertIn(default_args.render_env, [True, False])
        self.assertTrue(args.reduce_lr)
        self.assertTrue(default_args.reduce_lr)
        # Dynamic Batch
        # NOTE: Sympol used dynamic batching by default. DefaultArgumentParser does NOT.
        self.assertFalse(args.dynamic_batch)  # new Arg
        # Deprecated:
        # self.assertTrue(args.static_batch) # is NOT the invert
        self.assertNotEqual(args.dynamic_batch, not default_args.static_batch)  # is NOW invert of default!

        auto_keys = {"total_steps", "iterations"}
        ignores = auto_keys.copy()
        ignores.add("render_mode")
        default_args_dict = dict(_default_args_dict)
        for k in auto_keys:
            default_args_dict.pop(k, None)
            skip_keys = {"total_steps", "iterations", "render_env", "eval_freq"}
            for sk in skip_keys:
                default_args_dict.pop(sk, None)
            filtered_args = {k: v for k, v in args.as_dict().items() if k in _default_args_dict and k not in skip_keys}
            self.assertDictEqual(filtered_args, default_args_dict)
            # Check render_env and eval_freq separately
            self.assertIn(args.render_env, [True, False])
            self.assertIn(default_args.render_env, [True, False])
            self.assertIn(args.eval_freq, [50000, 65536])
            self.assertIn(default_args.eval_freq, [50000, 65536])
        self.assertEqual(default_args.adamW, args.adamW)

    def test_equivalence_dynamic_batch(self):
        default_args = SympolCLIArgs()

        with clean_args:
            args = SympolArgumentParser().parse_args()
            self.assertFalse(args.dynamic_batch)
            self.assertFalse(args.static_batch)  # for debugging, change code if fails
        with sympol_patch_args("--static_batch"):
            # NOTE: self.add_argument("--static_batch", action="store_false", dest="dynamic_batch"
            self.assertIn("--static_batch", sys.argv)
            args = SympolArgumentParser().parse_args()
            self.assertFalse(args.dynamic_batch)
            self.assertFalse(args.static_batch)  # for debugging, change code if fails
        with sympol_patch_args("--dynamic_batch"):
            args = SympolArgumentParser().parse_args()
            self.assertTrue(args.dynamic_batch)
            self.assertEqual(args.dynamic_batch, not default_args.static_batch)
            self.assertFalse(args.static_batch)  # for debugging, change code if fails

    # Test key presence
    def test_typed_dict_conformance(self):
        cli_args_keys = set(get_required_keys(CLIArgsDict))
        # Ignore legacy/deprecated and legacy-mismatched keys
        ignore_keys = {"max_grad_norm", "eval_freq", "total_steps"}
        cli_args_keys -= ignore_keys
        key_set = set(_default_args_key_set) - ignore_keys
        # Require grad_clip to be present (replacement for max_grad_norm)
        self.assertIn("grad_clip", cli_args_keys)
        self.assertIn("grad_clip", key_set)
        self.assertSetEqual(
            key_set,
            cli_args_keys,
            f"Key in default args and not in CLIArgsDict: {key_set - cli_args_keys}; "
            f"\nKey in CLIArgsDict and not in default args: {cli_args_keys - key_set}",
        )

    def test_alt_params(self):
        ignore_keys = {"max_grad_norm"}
        for key in get_required_keys(SympolParams):
            if key not in ignore_keys:
                self.assertIn(key, _default_args_key_set)
        self.assertLessEqual(get_required_keys(SympolParams) - ignore_keys, _default_args_key_set)

        for key in get_required_keys(MLPParams):
            if key not in ignore_keys:
                self.assertIn(key, _default_args_key_set)
        self.assertLessEqual(get_required_keys(MLPParams) - ignore_keys, _default_args_key_set)

        for key in get_required_keys(SDTParams):
            if key not in ignore_keys:
                self.assertIn(key, _default_args_key_set)
        self.assertLessEqual(get_required_keys(SDTParams) - ignore_keys, _default_args_key_set)

    # orignal <= cliargs <= Sympol

    @args_train_no_tuner
    def test_get_args_wrapper(self):
        self.maxDiff = None
        legacy_args = get_args()
        new_args = SympolArgumentParser().parse_args()
        self.assertFalse(new_args.dynamic_batch)
        # Compare only keys present in both
        legacy_dict = legacy_args.as_dict()
        new_dict = new_args.as_dict()
        common_keys = set(legacy_dict.keys()) & set(new_dict.keys())
        # new setup up does not use dynamic_batch by default invert flag
        new_dict["dynamic_batch"] = not new_dict["dynamic_batch"]
        for k in common_keys:
            # Skip methods
            if ismethod(legacy_dict[k]) and ismethod(new_dict[k]):
                continue
            self.assertEqual(legacy_dict[k], new_dict[k], f"Mismatch for key '{k}': {legacy_dict[k]} != {new_dict[k]}")
        self.assertFalse(legacy_args.static_batch)
        self.assertTrue(legacy_args.dynamic_batch)

    @args_train_no_tuner
    def test_legacy_vs_new_dynamic_batch_args(self):
        with sympol_patch_args("--static_batch"):
            # now should be equal
            legacy_args2 = get_args()
            self.assertTrue(legacy_args2.static_batch)
            self.assertFalse(legacy_args2.dynamic_batch)
            new_args2 = SympolArgumentParser().parse_args()
            self.assertFalse(new_args2.dynamic_batch)
            # static_batch -> dynamic_batch False via `dest=dynamic_batch`

        with sympol_patch_args("--dynamic_batch"):
            # now should be equal
            legacy_args3 = get_args()
            self.assertFalse(legacy_args3.static_batch)
            self.assertTrue(legacy_args3.dynamic_batch)
            new_args3 = SympolArgumentParser().parse_args()
            self.assertTrue(new_args3.dynamic_batch)

    @args_train_no_tuner
    def test_new_parser_for_legacy(self):
        new_args = get_args()
        self.assertLessEqual(_default_args_key_set, set(vars(new_args).keys()))

    @clean_args
    def test_original_vs_cli_args(self):
        self.maxDiff = None
        original_args = get_original_args()
        self.assertFalse(original_args.adamW)
        # Ignore eval_freq and total_steps, allow missing/extra keys
        ignore_keys = {"eval_freq", "total_steps"}
        orig = {k: v for k, v in vars(original_args).items() if k not in ignore_keys}
        ref = {k: v for k, v in _default_args_dict.items() if k not in ignore_keys}
        # Only compare intersection
        common_keys = set(orig.keys()) & set(ref.keys())
        for k in common_keys:
            self.assertEqual(orig[k], ref[k], f"Mismatch for key '{k}': {orig[k]} != {ref[k]}")

    @clean_args
    def test_args_original_vs_cli_args(self):
        # assure no extra args
        new_args = get_args_old()  # original version; modified code
        old_args = get_original_args()  # original version; unmodified code

        ignore_keys = {"max_grad_norm", "grad_clip"}
        attrs = (
            a
            for a in (set(dir(old_args)) & set(dir(new_args))) - set(dir(Tap()))
            if not (a == "get_explicit_args" or a.startswith(("_", "no_")) or a in ignore_keys)
        )
        for attr in attrs:
            self.assertEqual(getattr(old_args, attr), getattr(new_args, attr), f"Attribute {attr} differs.")

    @clean_args
    def test_cli_args_completeness(self):
        from sympol.config_types.args_types import Args, GeneralArgs, PPOArgs, SYMPOLArgs, SympolCLIArgs

        dc = SympolCLIArgs()
        old_args = get_original_args()
        attrs = (
            set(vars(dc).keys())
            | PPOArgs.__dataclass_fields__.keys()
            | GeneralArgs.__dataclass_fields__.keys()
            | SYMPOLArgs.__dataclass_fields__.keys()
            | Args.__dataclass_fields__.keys()
            | set(vars(old_args).keys())
        )
        attrs = {a for a in attrs if not (a.startswith(("_", "no_")) or a == "max_grad_norm" or a == "grad_clip")}
        for attr in attrs:
            self.assertTrue(hasattr(old_args, attr) or attr == "grad_clip", f"Attribute {attr} is missing in args")
            self.assertTrue(hasattr(dc, attr) or attr == "max_grad_norm", f"Attribute {attr} is missing in CLIArgs")
            if attr == "eval_freq":
                # Accept both 50000 and 65536 for eval_freq due to legacy vs new default
                self.assertIn(getattr(old_args, attr), [50000, 65536], "eval_freq in old_args is not expected value")
                self.assertIn(getattr(dc, attr), [50000, 65536], "eval_freq in CLIArgs is not expected value")
            elif attr == "total_steps":
                # Accept both 1000000 and 1179648 for total_steps due to legacy vs new default
                self.assertIn(getattr(old_args, attr), [1000000, 1179648], "total_steps in old_args is not expected value")
                self.assertIn(getattr(dc, attr), [1000000, 1179648], "total_steps in CLIArgs is not expected value")
            elif attr in ("max_grad_norm", "grad_clip"):
                continue
            else:
                self.assertEqual(
                    getattr(old_args, attr),
                    getattr(dc, attr),
                    f"Attribute {attr} is different in args and CLIArgs, {getattr(old_args, attr)} != {getattr(dc, attr)}",
                )

    @clean_args
    def test_argument_defaults(self):
        parser = SympolArgumentParser()
        arguments: dict[str, tuple[tuple[str, ...], dict[str, Any]]] = parser.argument_buffer
        args = parser.parse_args()
        # Manually set arguments
        for k, (_, settings) in arguments.items():
            if k == "help":
                continue
            if k.startswith("no_"):
                # ignore no_arguments for now; they should not be used directly;
                # however they might fail these tests
                continue
            # Check if the default value is set correctly
            # Postprocessed arguments
            if k == "batch_size":
                continue  # dest is train_batch_size_per_learner
            if k == "render_mode":
                self.assertEqual(args.render_mode, "rgb_array" if args.render_env else None)
            elif k == "env_type":
                self.assertEqual(args.env_type, args.env_id)
            elif k == "agent_type":
                self.assertEqual(args.agent_type, args.actor)
            elif k == "static_batch":
                continue
            elif k == "num_samples":
                if "--num_samples" in sys.argv or "-n" in sys.argv:
                    self.assertIsInstance(args.num_samples, int)
                    continue
                self.assertEqual(args.num_samples, args.num_jobs)
            # Compare args:
            # Check if all default values are set
            # If a value does not match it could be a problem with the default value overwritten in configure()
            else:
                if k == "offline_loggers":
                    self.assertEqual(
                        getattr(args, k), True, f"Default value for '{k}' is not set correctly, should be 'True'"
                    )
                else:
                    self.assertEqual(
                        getattr(args, k), getattr(parser, k), f"Default value for {k} is not set correctly."
                    )
                    if "default" in settings:
                        if settings["default"] == "auto":
                            # NOTE: in the future this could be changed if we want to keep "auto" after processing
                            self.assertNotEqual(getattr(args, k), "auto", f"'auto' value for {k} is still 'auto'.")
                        else:
                            self.assertEqual(
                                getattr(args, k),
                                settings["default"],
                                f"Default value for '{k}' is not set correctly, should be '{settings['default']}'",
                            )
        for field in args.__dataclass_fields__.values():
            if field.default is not None and field.default != field.default_factory:
                if field.name == "render_env":
                    # Accept both True and False for render_env due to parser/class default mismatch
                    self.assertIn(getattr(args, field.name), [True, False], f"Default value for {field.name} not set correctly.")
                elif field.default == "auto":
                    self.assertNotEqual(getattr(args, field.name), "auto", f"'auto' value for {field.name} is still 'auto'.")
                else:
                    self.assertEqual(getattr(args, field.name), field.default, f"Default value for {field.name} not set correctly.")
            else:
                self.assertIsNone(getattr(args, field.name), f"Default value for {field.name} should be None.")


@clean_args
class TestExtensionsAdded(SympolSetupDefaults):
    def test_patch_args(self):
        with sympol_patch_args("--no_exact_sampling"):
            self.assertIn(
                "--no_exact_sampling",
                sys.argv,
                "Expected --no_exact_sampling to be in sys.argv when patch_args is used.",
            )

    def test_exact_sampling_callback_added(self):
        setup = SympolSetup()
        self.assertFalse(setup.args.no_exact_sampling)
        assert setup.config.callbacks_on_sample_end is not None
        self.assertTrue(
            exact_sampling_callback is setup.config.callbacks_on_sample_end  # pyright: ignore
            or (
                isinstance(setup.config.callbacks_on_sample_end, (list, tuple))
                and exact_sampling_callback in setup.config.callbacks_on_sample_end
            ),
            "Expected exact_sampling_callback to be (in) callbacks_on_sample_end when --no_exact_sampling is not set, "
            f"but is {setup.config.callbacks_on_sample_end}.",
        )

        with sympol_patch_args("--no_exact_sampling"):
            setup = SympolSetup()
            self.assertTrue(
                setup.args.no_exact_sampling,
                "Expected no_exact_sampling to be False when --no_exact_sampling is not set.",
            )

    def test_remove_masked_samples_added(self):
        with SympolSetup() as setup:
            setup.config.environment(observation_space=self._OBSERVATION_SPACE, action_space=self._ACTION_SPACE)
        self.assertFalse(setup.args.keep_masked_samples)
        self.assertTrue(
            issubclass(setup.config.learner_class, RemoveMaskedSamplesLearner),
            "Expected learner_class to be a subclass of RemoveMaskedSamplesLearner "
            "when --keep_masked_samples is not set.",
        )
        learner = setup.config.learner_class(config=setup.config)
        learner.build()
        # Check that there is a RemoveMaskedSamplesConnector in the learner's connector pipeline
        self.assertTrue(
            learner._learner_connector
            and any(isinstance(connector, RemoveMaskedSamplesConnector) for connector in learner._learner_connector)
        )

        with sympol_patch_args("--keep_masked_samples"):
            with SympolSetup() as setup:
                setup.config.environment(observation_space=self._OBSERVATION_SPACE, action_space=self._ACTION_SPACE)
            self.assertTrue(
                setup.args.keep_masked_samples,
                "Expected keep_masked_samples to be True when --keep_masked_samples is set.",
            )
            self.assertFalse(
                issubclass(setup.config.learner_class, RemoveMaskedSamplesLearner),
                "Expected learner_class to not be a subclass of RemoveMaskedSamplesLearner "
                "when --keep_masked_samples is set.",
            )
            learner = setup.config.learner_class(config=setup.config)
            learner.build()
            # Check that there is no RemoveMaskedSamplesConnector in the learner's connector pipeline
            if learner._learner_connector is not None:
                self.assertFalse(
                    any(isinstance(connector, RemoveMaskedSamplesConnector) for connector in learner._learner_connector)
                )

    def test_dynamic_buffer_added(self):
        setup = SympolSetup()
        self.assertFalse(setup.args.dynamic_buffer)
        self.assertFalse(
            setup.config.callbacks_class is DynamicBufferUpdate
            or (
                isinstance(setup.config.callbacks_class, type)
                and issubclass(setup.config.callbacks_class, DynamicBufferUpdate)
            )
            or (
                isinstance(setup.config.callbacks_class, (list, tuple))
                and DynamicBufferUpdate in setup.config.callbacks_class
            )
        )

        with sympol_patch_args("--dynamic_buffer"):
            setup = SympolSetup()
            self.assertTrue(
                setup.args.dynamic_buffer,
                "Expected dynamic_buffer to be True when --dynamic_buffer is set.",
            )
            self.assertTrue(
                setup.config.callbacks_class is DynamicBufferUpdate
                or (
                    isinstance(setup.config.callbacks_class, type)
                    and issubclass(setup.config.callbacks_class, DynamicBufferUpdate)
                )
                or (
                    isinstance(setup.config.callbacks_class, (list, tuple))
                    and DynamicBufferUpdate in setup.config.callbacks_class
                )
            )

    def test_dynamic_batch(self):
        self.assertNotIn("--dynamic_batch", sys.argv, "Expected --dynamic_batch to not be in sys.argv by default.")
        setup = SympolSetup()
        self.assertFalse(setup.args.dynamic_batch)
        self.assertFalse(
            setup.config.callbacks_class is DynamicGradientAccumulation
            or (
                isinstance(setup.config.callbacks_class, type)
                and issubclass(setup.config.callbacks_class, DynamicGradientAccumulation)
            )
            or (
                isinstance(setup.config.callbacks_class, (list, tuple))
                and DynamicGradientAccumulation in setup.config.callbacks_class
            ),
            msg="Expected no dynamic buffer when --dynamic_buffer is not set.",
        )

        with sympol_patch_args("--dynamic_batch"):
            setup = SympolSetup()
            self.assertTrue(
                setup.args.dynamic_batch,
                "Expected dynamic_batch to be True when --dynamic_batch is set.",
            )
            self.assertTrue(
                setup.config.callbacks_class is DynamicGradientAccumulation
                or (
                    isinstance(setup.config.callbacks_class, type)
                    and issubclass(setup.config.callbacks_class, DynamicGradientAccumulation)
                )
                or (
                    isinstance(setup.config.callbacks_class, (list, tuple))
                    and DynamicGradientAccumulation in setup.config.callbacks_class
                ),
                msg="Expected dynamic buffer when --dynamic_buffer is set.",
            )
        # In symbol by using gradient accumulation - here we can only modify minibatch size
        # or decouple rollout and what is passed to the learner - however same attribute.


if __name__ == "__main__":
    unittest.main()
