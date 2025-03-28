from dataclasses import asdict
import sys
import unittest
from unittest import mock

import jax

import args
from tests._test_utils import clean_args, fixed_args

from config_types.args_types import CLIArgs
from rllib_port.sympol.sympol_model import SympolRLModel
from rllib_port.sympol.sympol_module import SympolPPOModule
from sympol import SYMPOL_RL


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
        model = SympolRLModel(obs_dim=2, action_dim=1, config=asdict(CLIArgs()))  # pyright: ignore[reportArgumentType]
        init_state = model.init_state(jax.random.PRNGKey(0), jax.numpy.zeros((2, 2)))

    def test_setup_instantiation(self):
        # Test
        from rllib_port.sympol.sympol_setup import SympolSetup

        for actor in ["sympol", "sdt", "mlp"]:
            with mock.patch.object(sys, "argv", ["file.py", "--agent_type", actor]):
                SympolSetup()

    def test_sympol_underlying_model(self):
        config = asdict(CLIArgs())
        actor = SYMPOL_RL(
            obs_dim=2,
            action_dim=1,
            depth=config["depth"],
            n_estimators=config["n_estimators"],
            action_type=config["action_type"],
            subset_fraction=config.get("subset_fraction", 0.8),
        )
        actor.config = config  # type: ignore[assignment]
        actor_state = SympolRLModel.init_state(actor, jax.random.PRNGKey(0), jax.numpy.zeros((2, 2)))
        out = actor.apply(
            actor_state.params,
            jax.numpy.ones((2, 2)),
            indices=actor_state.indices,
        )

    def test_sympol_apply(self):
        model = SympolRLModel(obs_dim=2, action_dim=1, config=asdict(CLIArgs()))  # pyright: ignore[reportArgumentType]
        actor_state = model.init_state(jax.random.PRNGKey(0), jax.numpy.zeros((2, 2)))
        # indices =model.init_indices(jax.random.PRNGKey(0))
        model.apply(
            actor_state.params,
            jax.numpy.ones((2, 2)),
            indices=actor_state.indices,
        )

    def test_sympol_call(self):
        model = SympolRLModel(obs_dim=2, action_dim=1, config=asdict(CLIArgs()))  # pyright: ignore[reportArgumentType]
        actor_state = model.init_state(jax.random.PRNGKey(0), jax.numpy.zeros((2, 2)))
        # indices =model.init_indices(jax.random.PRNGKey(0))
        model({"params": actor_state.params, "obs": jax.numpy.ones((2, 2)), "indices": actor_state.indices})


if __name__ == "__main__":
    unittest.main()
