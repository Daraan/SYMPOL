from dataclasses import asdict
import sys
import unittest
from unittest import mock

from _test_utils import clean_args, fixed_args

from config_types.args_types import CLIArgs
from rllib_port.sympol.sympol_model import SympolRLModel
from rllib_port.sympol.sympol_module import SympolPPOModule


@clean_args
class TestModule(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if {"-v", "test*.py"} & set(sys.argv):
            print("disable breakpoint")
            mock.patch("builtins.breakpoint").start()
            mock.seal
        else:
            print("enable breakpoint")
        breakpoint()

    def test_module_setup(self):
        # Test
        import gymnasium as gym

        env = gym.make("CartPole-v1")

        from config_types.args_types import CLIArgs

        module = SympolPPOModule(
            observation_space=env.observation_space,
            action_space=env.action_space,
            model_config=asdict(CLIArgs()),
            inference_only=False,
        )
        module.setup()

    def test_sympol_creation(self):
        # test
        import jax

        model = SympolRLModel(obs_dim=0, action_dim=0, config=asdict(CLIArgs()))
        model({"obs": jax.numpy.array([1, 2, 3])})

    def test_setup(self):
        # Test
        from rllib_port.sympol.sympol_setup import SympolSetup

        for actor in ["sympol", "sdt", "mlp"]:
            with mock.patch.object(sys, "argv", ["file.py", "--agent_type", actor]):
                SympolSetup()


if __name__ == "__main__":
    unittest.main()
