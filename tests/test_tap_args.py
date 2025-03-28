import unittest

from args import get_args, get_args_old  # noqa: F401  # avoid circular imports
from config_types.args_types import CLIArgs
from config_types.params_types import CLIArgsDict, MLPParams, SDTParams, SympolParams
from rllib_port.extended_args import SympolArgumentParser

from tests._original_args import get_original_args
from tests._test_utils import clean_args, fixed_args, get_required_keys

_default_args = CLIArgs()
_default_args_dict = vars(_default_args)
_default_args_key_set = set(_default_args_dict.keys())


class TestArgs(unittest.TestCase):
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
