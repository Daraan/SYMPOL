from __future__ import annotations

import dataclasses
import os
import sys
import unittest
from typing import TYPE_CHECKING, cast
from unittest import mock

import gymnasium as gym
import jax
import numpy.testing as npt
from ray.rllib.core.columns import Columns

import args  # noqa: F401
from mlp import Critic_MLP
from ray_utilities.callbacks.algorithm.dynamic_buffer_callback import DynamicBufferUpdate
from rllib_port.core.sympol_module import SympolPPOModule
from rllib_port.mlp.mlp_model import ActorMLPModel, CriticMLPModel
from rllib_port.sdt.sdt_model import ActorSDTModel, CriticSDTModel
from rllib_port.sympol.sympol_model import SympolRLModel
from rllib_port.sympol_setup import SympolSetup
from sympol import SYMPOL_RL
from tests._test_utils import DisableBreakpointsForGUI, SetupDefaults, clean_args

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.connectors.env_to_module.env_to_module_pipeline import EnvToModulePipeline

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
            out2 = model({"obs": self._DEFAULT_INPUT}, parameters=actor_state2.params, indices=actor_state2.indices)

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
                out = model({"obs": self._DEFAULT_INPUT}, parameters=init_state.params)
                npt.assert_array_almost_equal(out, o_out, decimal=5)  # type: ignore

    def test_critic_mlp(self):
        from rllib_port.mlp.mlp_model import CriticMLPModel

        model = CriticMLPModel(self._DEFAULT_CONFIG_DICT)
        with self.subTest("original_critic"):
            o_critic = Critic_MLP()
            o_critic_state = self._create_critic_state(o_critic)

            o_out = o_critic.apply(o_critic_state.params, self._DEFAULT_INPUT)

        init_state = model.init_state(self._CRITIC_KEY, self._ENV_SAMPLE)
        out = model.__call__({"obs": self._DEFAULT_INPUT}, parameters=init_state.params)
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
            critic_out = module.compute_values({"obs": self._DEFAULT_INPUT}, parameters=None)
            # training:
            critic_out2 = module.compute_values({"obs": self._DEFAULT_INPUT}, parameters=module.states["critic"].params)
            npt.assert_array_almost_equal(critic_out, critic_out2, decimal=5)  # type: ignore

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

    def test_indices_frozen(self):
        indices = SYMPOL_RL(
            obs_dim=self._OBS_DIM,
            action_dim=self._ACTION_DIM,
            depth=self._DEFAULT_NAMESPACE.depth,
            n_estimators=self._DEFAULT_NAMESPACE.n_estimators,
            action_type=self._DEFAULT_NAMESPACE.action_type,
            subset_fraction=0.8,
        ).init_indices(self._ACTOR_KEY)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            indices.features_by_estimator = indices.features_by_estimator.at[-1, -1].add(1)  # pyright: ignore[reportAttributeAccessIssue] # readonly

    def test_indices_equivalence(self):
        modelA1 = SYMPOL_RL(
            obs_dim=self._OBS_DIM,
            action_dim=self._ACTION_DIM,
            depth=self._DEFAULT_NAMESPACE.depth,
            n_estimators=self._DEFAULT_NAMESPACE.n_estimators,
            action_type=self._DEFAULT_NAMESPACE.action_type,
            subset_fraction=0.8,
        )
        modelA2 = SYMPOL_RL(
            obs_dim=self._OBS_DIM,
            action_dim=self._ACTION_DIM,
            depth=self._DEFAULT_NAMESPACE.depth,
            n_estimators=self._DEFAULT_NAMESPACE.n_estimators,
            action_type=self._DEFAULT_NAMESPACE.action_type,
            subset_fraction=0.8,
        )

        # Test repeated use
        indicesA11 = modelA1.init_indices(self._ACTOR_KEY)
        indicesA12 = modelA1.init_indices(self._ACTOR_KEY)
        indicesA2 = modelA2.init_indices(self._ACTOR_KEY)
        self.assertEqual(indicesA11, indicesA12)
        self.assertEqual(hash(indicesA11), hash(indicesA12))
        self.assertEqual(hash(indicesA11), hash(indicesA2))
        self.assertEqual(indicesA11, indicesA2)

        indicesA1_other = modelA1.init_indices(self._CRITIC_KEY)
        self.assertNotEqual(indicesA11, indicesA1_other)
        self.assertNotEqual(hash(indicesA11), hash(indicesA1_other))
        # Only features are different:
        npt.assert_array_equal(indicesA11.internal_node_index_list, indicesA1_other.internal_node_index_list)
        npt.assert_array_equal(indicesA11.path_identifier_list, indicesA1_other.path_identifier_list)
        with self.assertRaises(AssertionError):
            npt.assert_array_equal(
                indicesA11.features_by_estimator,
                indicesA1_other.features_by_estimator,
            )

        # Meta test:
        def _calc_selected_variable(subset_fraction):
            if self._DEFAULT_NAMESPACE.n_estimators > 1:
                selected_variables = int(self._OBS_DIM * subset_fraction)
                selected_variables = min(selected_variables, 50)
                selected_variables = max(selected_variables, 10)
                selected_variables = min(selected_variables, self._OBS_DIM)
                if not selected_variables * self._DEFAULT_NAMESPACE.n_estimators > 3 * self._OBS_DIM:
                    selected_variables = self._OBS_DIM
            else:
                selected_variables = self._OBS_DIM

        if False:
            self.assertNotEqual(
                _calc_selected_variable(0.8),
                _calc_selected_variable(2.0),
                "To get a different output, the subset fraction must be different AND n_estimators > 1!",
            )
            modelB = SYMPOL_RL(
                obs_dim=self._OBS_DIM,
                action_dim=self._ACTION_DIM,
                depth=self._DEFAULT_NAMESPACE.depth,
                n_estimators=self._DEFAULT_NAMESPACE.n_estimators,
                action_type=self._DEFAULT_NAMESPACE.action_type,
                subset_fraction=2.0,
            )
            # only features are affected by random key
            indicesB = modelB.init_indices(self._ACTOR_KEY)
            npt.assert_array_equal(indicesA11.features_by_estimator, indicesB.features_by_estimator)
            npt.assert_array_equal(indicesA11.internal_node_index_list, indicesB.internal_node_index_list)

            self.assertNotEqual(indicesA11, indicesB)
            self.assertNotEqual(hash(indicesA11), hash(indicesB))

        indicesC = SYMPOL_RL(
            obs_dim=self._OBS_DIM,
            action_dim=self._ACTION_DIM,
            depth=self._DEFAULT_NAMESPACE.depth + 1,
            n_estimators=self._DEFAULT_NAMESPACE.n_estimators,
            action_type=self._DEFAULT_NAMESPACE.action_type,
            subset_fraction=0.8,
        ).init_indices(self._ACTOR_KEY)
        self.assertNotEqual(indicesA11, indicesC)
        self.assertNotEqual(hash(indicesA11), hash(indicesC))
        if False:
            self.assertNotEqual(indicesC, indicesB)
            self.assertNotEqual(hash(indicesC), hash(indicesB))


class TestSetup(DisableBreakpointsForGUI, SetupDefaults):
    def test_setup_instantiation(self):
        for actor in ["sympol", "sdt", "mlp"]:
            with mock.patch.object(sys, "argv", ["file.py", "--agent_type", actor]):
                # fails as expects trial parameter
                SympolSetup()

    def test_dynamic_buffer_callback(self):
        # Adding dynamic buffer
        with mock.patch.object(sys, "argv", ["file.py", "--dynamic_buffer"]):
            # fails as expects trial parameter
            config = SympolSetup().config
            assert (
                config.callbacks_class is DynamicBufferUpdate
                or (isinstance(config.callbacks_class, list) and DynamicBufferUpdate in config.callbacks_class)
                or (
                    getattr(config.callbacks_class, "IS_CALLBACK_CONTAINER", False)
                    and DynamicBufferUpdate in config.callbacks_class._callback_list  # type: ignore[attr-defined]
                )
            )
        # adding no buffer
        with mock.patch.object(sys, "argv", ["file.py"]):
            # fails as expects trial parameter
            config = SympolSetup().config
            assert (
                config.callbacks_class is not DynamicBufferUpdate
                and (not isinstance(config.callbacks_class, list) or DynamicBufferUpdate not in config.callbacks_class)
                and (
                    not getattr(config.callbacks_class, "IS_CALLBACK_CONTAINER", False)
                    or DynamicBufferUpdate not in config.callbacks_class._callback_list  # type: ignore[attr-defined]
                )
            )


if __name__ == "__main__":
    unittest.main()
