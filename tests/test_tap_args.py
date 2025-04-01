import unittest
import unittest.mock

from args import get_args, get_args_old  # noqa: F401  # avoid circular imports
from config_types.args_types import CLIArgs
from dataclasses import asdict
from config_types.params_types import CLIArgsDict, MLPParams, SDTParams, SympolParams
from rllib_port.extended_args import SympolArgumentParser

from rllib_port.sympol.sympol_setup import SympolSetup
from tests._original_args import get_original_args
from tests._test_utils import clean_args, fixed_args, get_required_keys

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
        parser = SympolArgumentParser()
        args = parser.parse_args()
        # CLIArgs <= SympolArgumentParser
        self.assertFalse(CLIArgs().adamW)
        self.assertFalse(args.adamW)  # why is this True
        self.assertDictEqual(
            {k: v for k, v in args.as_dict().items() if k in _default_args_dict},
            _default_args_dict,
        )
        self.assertEqual(CLIArgs().adamW, args.adamW)

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

    @fixed_args
    def test_get_args_wrapper(self):
        self.assertEqual(get_args(), SympolArgumentParser().parse_args())

    @fixed_args
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


if __name__ == "__main__":
    unittest.main()
