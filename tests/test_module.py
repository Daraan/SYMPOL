from dataclasses import asdict
import sys
import unittest
from unittest import mock

import jax

import args  # noqa: F401
from tests._test_utils import DisableBreakpointsForGUI, clean_args, fixed_args

from config_types.args_types import CLIArgs
from rllib_port.sympol.sympol_model import SympolRLModel
from rllib_port.sympol.sympol_module import SympolPPOModule
from sympol import SYMPOL_RL
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ray_utilities.jax.jax_model import PureJaxModelProtocol


@clean_args
class TestModels(DisableBreakpointsForGUI):
    def test_sympol_creation(self):
        model = SympolRLModel(obs_dim=2, action_dim=1, config=asdict(CLIArgs()))  # pyright: ignore[reportArgumentType]
        init_state = model.init_state(jax.random.PRNGKey(0), jax.numpy.zeros((2, 2)))

    def test_old_model_vs_module(self):
        with self.subTest("test_sympol_underlying_model"):
            config1 = asdict(CLIArgs())
            actor: PureJaxModelProtocol = SYMPOL_RL(
                obs_dim=2,
                action_dim=1,
                depth=config1["depth"],
                n_estimators=config1["n_estimators"],
                action_type=config1["action_type"],
                subset_fraction=config1.get("subset_fraction", 0.8),
            )
            actor_state1 = SympolRLModel.init_state(
                actor,
                jax.random.PRNGKey(0),
                jax.numpy.zeros((2, 2)),
                config=config1,  # type: ignore[reportArgumentType]
            )
            out1 = actor.apply(
                actor_state1.params,
                jax.numpy.ones((2, 2)),
                indices=actor_state1.indices,
            )

        with self.subTest("config unmodified"):
            config2 = asdict(CLIArgs())
            self.assertEqual(config1, config2)

        with self.subTest("test_sympol_call"):
            model = SympolRLModel(obs_dim=2, action_dim=1, config=config2)  # pyright: ignore[reportArgumentType]
            actor_state2 = model.init_state(jax.random.PRNGKey(0), jax.numpy.zeros((2, 2)))
            out2 = model(
                {"params": actor_state2.params, "obs": jax.numpy.ones((2, 2)), "indices": actor_state2.indices}
            )

        with self.subTest("test_sympol_output_shape"):
            self.assertEqual(out1.shape, out2.shape)
            self.assertEqual(out1.shape, (2, 1))
            self.assertTrue(jax.numpy.array_equal(out1, out2))


class TestModuleAndSetup(DisableBreakpointsForGUI):
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

    def test_setup_instantiation(self):
        # Test
        from rllib_port.sympol.sympol_setup import SympolSetup

        for actor in ["sympol", "sdt", "mlp"]:
            with mock.patch.object(sys, "argv", ["file.py", "--agent_type", actor]):
                SympolSetup()


if __name__ == "__main__":
    unittest.main()
