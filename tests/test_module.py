from __future__ import annotations

import os
import sys
import unittest
from typing import TYPE_CHECKING
from unittest import mock

import jax
import numpy.testing as npt
from ray.rllib.core.columns import Columns
import args  # noqa: F401

from mlp import Critic_MLP
from rllib_port.mlp.mlp_model import ActorMLPModel, CriticMLPModel
from rllib_port.sdt.sdt_model import ActorSDTModel, CriticSDTModel
from rllib_port.sympol.sympol_model import SympolRLModel
from rllib_port.sympol.sympol_module import SympolPPOModule
from rllib_port.sympol.sympol_setup import SympolSetup
from sympol import SYMPOL_RL
from tests._test_utils import DisableBreakpointsForGUI, SetupDefaults, clean_args

if TYPE_CHECKING:
    from ray_utilities.jax.jax_model import PureJaxModelProtocol


os.environ["RAY_DEBUG"] = "0"


@clean_args
class TestModels(DisableBreakpointsForGUI, SetupDefaults):
    def test_sympol_creation(self):
        model = SympolRLModel(obs_dim=2, action_dim=self._ACTION_DIM, config=self._DEFAULT_CONFIG_DICT)  # pyright: ignore[reportArgumentType]
        model.init_state(self._ACTOR_KEY, jax.numpy.zeros((2, 2)))

    def test_old_model_vs_module(self):
        with self.subTest("test_sympol_underlying_model"):
            config1 = self._DEFAULT_CONFIG_DICT
            actor: PureJaxModelProtocol = SYMPOL_RL(
                obs_dim=2,
                action_dim=self._ACTION_DIM,
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
            model = SympolRLModel(obs_dim=2, action_dim=self._ACTION_DIM, config=config2)  # pyright: ignore[reportArgumentType]
            actor_state2 = model.init_state(self._ACTOR_KEY, self._ENV_SAMPLE)
            out2 = model({"state": actor_state2, "obs": self._DEFAULT_INPUT})

            npt.assert_allclose(out1, out2)

    def test_old_mlp_sdt_vs_new(self):
        for func, model_cls in zip(
            [self._create_actor_mlp, self._create_critic_mlp, self._create_actor_sdt, self._create_critic_sdt],
            [ActorMLPModel, CriticMLPModel, ActorSDTModel, CriticSDTModel],
        ):
            with self.subTest("Compare model out", func=func.__name__, model_cls=model_cls.__name__):
                o_model, o_state = func()

                if model_cls in (CriticMLPModel, CriticSDTModel):
                    o_out = o_model.apply(
                        o_state.params,
                        self._DEFAULT_INPUT,
                    )
                    model = model_cls(self._DEFAULT_CONFIG_DICT)
                    init_state = model.init_state(self._CRITIC_KEY, self._ENV_SAMPLE)
                else:
                    o_out = o_model.apply(
                        o_state.params,
                        self._DEFAULT_INPUT,
                        indices=o_state.indices,  # pyright: ignore[reportAttributeAccessIssue]
                    )
                    model = model_cls(self._DEFAULT_CONFIG_DICT, action_dim=self._ACTION_DIM)
                    init_state = model.init_state(self._ACTOR_KEY, self._ENV_SAMPLE)
                out = model({"state": init_state, "obs": self._DEFAULT_INPUT})
                npt.assert_array_almost_equal(out, o_out, decimal=5)  # type: ignore

    def test_critic_mlp(self):
        from rllib_port.mlp.mlp_model import CriticMLPModel

        model = CriticMLPModel(self._DEFAULT_CONFIG_DICT)
        with self.subTest("original_critic"):
            o_critic = Critic_MLP()
            o_critic_state = self._create_critic_state(o_critic)

            o_out = o_critic.apply(o_critic_state.params, self._DEFAULT_INPUT)

        init_state = model.init_state(self._CRITIC_KEY, self._ENV_SAMPLE)
        out = model.__call__({"state": init_state, "obs": self._DEFAULT_INPUT})
        npt.assert_array_almost_equal(out, o_out, decimal=5)  # type: ignore


class TestSympolModule(DisableBreakpointsForGUI, SetupDefaults):
    def test_module_setup(self):
        # Test
        module = SympolPPOModule(
            observation_space=self._OBSERVATION_SPACE,
            action_space=self._ACTION_SPACE,
            model_config=self._DEFAULT_CONFIG_DICT,
            inference_only=False,
        )
        module.setup()

    def test_state_equivalence(self):
        module = SympolPPOModule(
            observation_space=self._OBSERVATION_SPACE,
            action_space=self._ACTION_SPACE,
            model_config=self._DEFAULT_CONFIG_DICT,
            inference_only=False,
        )
        module.setup()
        _, o_state = self._create_critic_mlp()
        self.util_test_state_equivalence(module.states["critic"], o_state)
        _, o_actor_state = self._create_actor_sympol()
        self.util_test_state_equivalence(module.states["actor"], o_actor_state)

    def test_output_equivalence(self):
        module = SympolPPOModule(
            observation_space=self._OBSERVATION_SPACE,
            action_space=self._ACTION_SPACE,
            model_config=self._DEFAULT_CONFIG_DICT,
            inference_only=False,
        )
        module.setup()
        self.assertEqual(module.model_config["critic"], "mlp")
        with self.subTest("test critic", critic=module.model_config["critic"]):
            critic_out = module.compute_values({"obs": self._DEFAULT_INPUT, "state": module.states["critic"]})
            # has a squeeze(-1) at the end, reverse:
            critic_out = critic_out[..., None]  # NOTE: Output shape is not the same

            o_model, o_state = self._create_critic_mlp()
            o_critic_out = o_model.apply(o_state.params, self._DEFAULT_INPUT)
            npt.assert_array_almost_equal(o_critic_out, critic_out, decimal=5)  # type: ignore

        self.assertEqual(module.model_config["actor"], "sympol")
        with self.subTest("test actor", actor=module.model_config["actor"]):
            o_actor, o_actor_state = self._create_actor_sympol()

            actor_output = module._forward({"obs": self._DEFAULT_INPUT, "state": module.states["actor"]})
            actor_out = actor_output[Columns.ACTION_DIST_INPUTS]
            o_actor_out = o_actor.apply(o_actor_state.params, self._DEFAULT_INPUT, indices=o_actor_state.indices)
            npt.assert_array_almost_equal(actor_out, o_actor_out, decimal=5)


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
