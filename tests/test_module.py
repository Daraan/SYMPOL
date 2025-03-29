from __future__ import annotations

import os
import sys
import unittest
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Final
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

import args  # noqa: F401
from config_types.args_types import CLIArgs
from mlp import Actor_MLP, Critic_MLP
from rllib_port.mlp.mlp_model import ActorMLPModel, CriticMLPModel
from rllib_port.sdt.sdt_model import ActorSDTModel, CriticSDTModel
from rllib_port.sympol.sympol_model import SympolRLModel
from rllib_port.sympol.sympol_module import SympolPPOModule
from rllib_port.sympol.sympol_setup import SympolSetup
from sdt import Actor_SDT, Critic_SDT
from sympol import SYMPOL_RL
from tests._test_utils import DisableBreakpointsForGUI, SetupDefaults, clean_args, fixed_args
from utils.utils import ActorTrainState, TrainState
from types import MappingProxyType

if TYPE_CHECKING:
    from ray_utilities.jax.jax_model import PureJaxModelProtocol


os.environ["RAY_DEBUG"] = "0"


@clean_args
class TestModels(DisableBreakpointsForGUI, SetupDefaults):
    def test_sympol_creation(self):
        model = SympolRLModel(obs_dim=2, action_dim=1, config=self._DEFAULT_CONFIG_DICT)  # pyright: ignore[reportArgumentType]
        init_state = model.init_state(self._ACTOR_KEY, jax.numpy.zeros((2, 2)))

    def test_old_model_vs_module(self):
        with self.subTest("test_sympol_underlying_model"):
            config1 = self._DEFAULT_CONFIG_DICT
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
                self._ACTOR_KEY,
                self._ENV_SAMPLE,
                config=config1,  # type: ignore[reportArgumentType]
            )
            out1 = actor.apply(
                actor_state1.params,
                self._DEFAULT_INPUT,
                indices=actor_state1.indices,
            )

        with self.subTest("config unmodified"):
            config2 = self._DEFAULT_CONFIG_DICT
            self.assertEqual(config1, config2)

        with self.subTest("test_sympol_call"):
            model = SympolRLModel(obs_dim=2, action_dim=1, config=config2)  # pyright: ignore[reportArgumentType]
            actor_state2 = model.init_state(self._ACTOR_KEY, self._ENV_SAMPLE)
            out2 = model({"params": actor_state2.params, "obs": self._DEFAULT_INPUT, "indices": actor_state2.indices})

        npt.assert_allclose(out1, out2)

    def test_old_mlp_sdt_vs_new(self):
        import optax

        for func, model_cls in zip(
            [self._create_actor_mlp, self._create_critic_mlp, self._create_actor_sdt, self._create_critic_sdt],
            [ActorMLPModel, CriticMLPModel, ActorSDTModel, CriticSDTModel],
        ):
            with self.subTest("Compare model out", func=func.__name__, model_cls=model_cls.__name__):
                o_model = func()
                o_state = ActorTrainState.create(
                    apply_fn=None,
                    params=o_model.init(self._ACTOR_KEY, jnp.array([self._ENV_SAMPLE])),
                    tx=optax.chain(
                        optax.clip_by_global_norm(self._DEFAULT_NAMESPACE.max_grad_norm),
                        optax.inject_hyperparams(optax.adam)(self._DEFAULT_NAMESPACE.learning_rate_actor),
                    ),
                    grad_accum=jax.tree.map(
                        jnp.zeros_like, o_model.init(self._ACTOR_KEY, jnp.array([self._ENV_SAMPLE]))
                    ),
                    indices=None,
                )
                o_out = o_model.apply(
                    o_state.params,
                    self._DEFAULT_INPUT,
                )

                if model_cls in (CriticMLPModel, CriticSDTModel):
                    model = model_cls(self._DEFAULT_CONFIG_DICT)
                else:
                    model = model_cls(self._DEFAULT_CONFIG_DICT, action_dim=1)
                init_state = model.init_state(self._ACTOR_KEY, self._ENV_SAMPLE)
                out = model({"state": init_state, "obs": self._DEFAULT_INPUT})
                npt.assert_array_almost_equal(out, o_out, decimal=5)  # type: ignore

    def test_critic_mlp(self):
        from rllib_port.mlp.mlp_model import CriticMLPModel

        model = CriticMLPModel(self._DEFAULT_CONFIG_DICT)
        sample = jnp.eye(2, 2)
        input = jnp.eye(2, 2).T
        with self.subTest("original_critic"):
            import optax

            o_critic = Critic_MLP()
            critic_state = TrainState.create(
                apply_fn=None,
                params=o_critic.init(self._CRITIC_KEY, jnp.array([sample])),
                tx=optax.chain(
                    optax.clip_by_global_norm(self._DEFAULT_NAMESPACE.max_grad_norm),
                    # adam or adamW:
                    (optax.adamw if self._DEFAULT_NAMESPACE.adamW else optax.adam)(
                        learning_rate=self._DEFAULT_NAMESPACE.learning_rate_critic
                    ),
                ),
            )
            o_out = o_critic.apply(critic_state.params, input)
        self.assertTrue(jnp.array_equal(sample, jnp.eye(2, 2)))

        init_state = model.init_state(self._CRITIC_KEY, sample)
        out = model.__call__({"state": init_state, "obs": input})
        self.assertTrue(jnp.allclose(o_out, out).item())  # type: ignore


class TestSympolModule(DisableBreakpointsForGUI, SetupDefaults):
    def test_module_setup(self):
        # Test
        import gymnasium as gym

        env = gym.make("CartPole-v1")

        module = SympolPPOModule(
            observation_space=env.observation_space,
            action_space=env.action_space,
            model_config=self._DEFAULT_CONFIG_DICT,
            inference_only=False,
        )
        module.setup()


class TestSetup(DisableBreakpointsForGUI, SetupDefaults):
    def test_setup_instantiation(self):
        for actor in ["sympol", "sdt", "mlp"]:
            with mock.patch.object(sys, "argv", ["file.py", "--agent_type", actor]):
                SympolSetup()

    def Xtest_algorithm_build(self):
        with mock.patch.object(sys, "argv", ["file.py", "--agent_type", "sympol"]):
            setup = SympolSetup()
            algorithm_config = setup.config
            algorithm_config.framework("torch")
            algorithm_config.build_algo()


if __name__ == "__main__":
    unittest.main()
