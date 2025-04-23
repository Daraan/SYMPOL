import unittest
import unittest.mock
from dataclasses import asdict
from typing import Any

from args import get_args, get_args_old  # noqa: F401  # avoid circular imports
from config_types.args_types import CLIArgs
from config_types.params_types import CLIArgsDict, MLPParams, SDTParams, SympolParams
from rllib_port.extended_args import SympolArgumentParser
from rllib_port.sympol_setup import SympolSetup
from tests._original_args import get_original_args
from tests._test_utils import args_train_no_tuner, clean_args, get_required_keys

_default_args = CLIArgs()
# NOTE: In vars no_ attributes are removed; with asdict not!
_default_args_dict = vars(_default_args)
_default_args_key_set = set(_default_args_dict.keys())


class TestArgObjects(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.maxDiff = 2000

    def test_equivalence_cli_args(self):
        self.maxDiff = 2000
        self.assertEqual(hash(_default_args), hash(CLIArgs()))

        args_same = CLIArgs()
        args_same_dict = asdict(args_same)

        with self.assertRaises(AssertionError):
            # no_ attributes are not present when using vars; with asdict they are
            self.assertEqual(args_same_dict, _default_args_dict)

        same_dict_positive = {k: v for k, v in args_same_dict.items() if not k.startswith("no_")}
        self.assertDictEqual(_default_args_dict, same_dict_positive)
        self.assertEqual(hash(tuple(_default_args_dict.items())), hash(tuple(same_dict_positive.items())))

        # different hash

        args_different = CLIArgs()
        args_different.depth = 2
        args_different_dict = vars(args_different)
        self.assertNotEqual(args_different, _default_args)
        self.assertNotEqual(hash(args_different), hash(_default_args))
        self.assertNotEqual(args_different_dict, _default_args_dict)

        # difference in non-hashable attribute
        args_different_but_same = CLIArgs()
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
        self.maxDiff = 2000

    @clean_args
    def test_equivalence_tap(self):
        """
        NOTE: This test is runtime dependent on the order of execution of the attributes

        As it is NOT deterministic this test could pass or fail.
        e.g. if adamW; no_adamW is False
        """
        parser = SympolArgumentParser()
        args = parser.parse_args()
        # CLIArgs <= SympolArgumentParser
        default_args = CLIArgs()
        # no - attributes
        self.assertFalse(default_args.adamW)
        self.assertFalse(args.adamW)
        self.assertTrue(args.render_env)
        self.assertTrue(args.reduce_lr)
        self.assertTrue(default_args.render_env)
        self.assertTrue(default_args.reduce_lr)
        auto_keys = {"total_steps", "iterations"}
        default_args_dict = _default_args_dict.copy()
        for k in auto_keys:
            default_args_dict.pop(k, None)
        self.assertDictEqual(
            {k: v for k, v in args.as_dict().items() if k in _default_args_dict and k not in auto_keys},
            default_args_dict,
        )
        self.assertEqual(default_args.adamW, args.adamW)

    # Test key presence
    def test_typed_dict_conformance(self):
        self.assertEqual(
            _default_args_key_set,
            get_required_keys(CLIArgsDict),
        )

    def test_alt_params(self):
        for key in get_required_keys(SympolParams):
            self.assertIn(key, _default_args_key_set)
        self.assertLessEqual(get_required_keys(SympolParams), _default_args_key_set)

        for key in get_required_keys(MLPParams):
            self.assertIn(key, _default_args_key_set)
        self.assertLessEqual(get_required_keys(MLPParams), _default_args_key_set)

        for key in get_required_keys(SDTParams):
            self.assertIn(key, _default_args_key_set)
        self.assertLessEqual(get_required_keys(SDTParams), _default_args_key_set)

    # orignal <= cliargs <= Sympol

    @args_train_no_tuner
    def test_get_args_wrapper(self):
        self.assertEqual(get_args(), SympolArgumentParser().parse_args())

    @args_train_no_tuner
    def test_new_parser(self):
        new_args = get_args()
        self.assertLessEqual(_default_args_key_set, set(vars(new_args).keys()))

    @clean_args
    def test_original_vs_cli_args(self):
        original_args = get_original_args()
        self.assertFalse(original_args.adamW)
        self.assertEqual(vars(original_args), _default_args_dict)

    @clean_args
    def test_args_original_vs_cli_args(self):
        # assure no extra args
        new_args = get_args_old()
        old_args = get_original_args()

        from tap import Tap

        for attr in (
            a
            for a in (set(dir(old_args)) | set(dir(new_args))) - set(dir(Tap()))
            if not (a == "get_explicit_args" or a.startswith(("_", "no_")))
        ):
            self.assertEqual(getattr(old_args, attr), getattr(new_args, attr), f"Attribute {attr} differs.")

    @clean_args
    def test_cli_args_completeness(self):
        from config_types.args_types import Args, CLIArgs, GeneralArgs, PPOArgs, SYMPOLArgs

        dc = CLIArgs()
        old_args = get_original_args()
        attrs = (
            set(vars(dc).keys())
            | PPOArgs.__dataclass_fields__.keys()
            | GeneralArgs.__dataclass_fields__.keys()
            | SYMPOLArgs.__dataclass_fields__.keys()
            | Args.__dataclass_fields__.keys()
            | set(vars(old_args).keys())
        )
        attrs = {a for a in attrs if not (a.startswith(("_", "no_")))}
        for attr in attrs:
            self.assertTrue(hasattr(old_args, attr), f"Attribute {attr} is missing in args")
            self.assertTrue(hasattr(dc, attr), f"Attribute {attr} is missing in CLIArgs")
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
            # Compare args:
            # Check if all default values are set
            # If a value does not match it could be a problem with the default value overwritten in configure()
            else:
                self.assertEqual(getattr(args, k), getattr(parser, k), f"Default value for {k} is not set correctly.")
                if "default" in settings:
                    if settings["default"] == "auto":
                        # NOTE: in the future this could be changed if we want to keep "auto" after processing
                        self.assertNotEqual(getattr(args, k), "auto", f"'auto' value for {k} is still 'auto'.")
                    else:
                        self.assertEqual(
                            getattr(args, k), settings["default"], f"Default value for {k} is not set correctly."
                        )
        for field in args.__dataclass_fields__.values():
            if field.default is not None and field.default != field.default_factory:
                if field.default == "auto":
                    # NOTE: in the future this could be changed if we want to keep "auto" after processing
                    self.assertNotEqual(
                        getattr(args, field.name), "auto", f"'auto' value for {field.name} is still 'auto'."
                    )
                else:
                    self.assertEqual(
                        getattr(args, field.name), field.default, f"Default value for {field.name} not set correctly."
                    )
            else:
                self.assertIsNone(getattr(args, field.name), f"Default value for {field.name} should be None.")


if __name__ == "__main__":
    unittest.main()
