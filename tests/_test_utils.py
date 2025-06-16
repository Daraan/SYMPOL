from __future__ import annotations

import sys
import unittest
from dataclasses import asdict
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Collection
from unittest import mock

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy.testing as npt
import optax
import tree
from typing_extensions import NotRequired, Required, get_origin, get_type_hints

from config_types.args_types import SympolCLIArgs
from mlp import Actor_MLP, Critic_MLP
from sdt import Actor_SDT, Critic_SDT
from sympol import SYMPOL_RL
from utils.utils import ActorTrainState, TrainState

if TYPE_CHECKING:
    import chex

args_train_no_tuner = mock.patch.object(sys, "argv", ["file.py", "--no-render_env", "-J", "1", "-it", "2", "-np"])
clean_args = mock.patch.object(sys, "argv", ["file.py"])
"""Use when comparing to CLIArgs"""


def patch_args(*args):
    """Patch sys.argv with the given args."""
    return mock.patch.object(sys, "argv", ["file.py", *args])


def get_explicit_required_keys(cls):
    return {k for k, v in get_type_hints(cls, include_extras=True).items() if get_origin(v) is Required}


def get_explicit_unrequired_keys(cls):
    return {k for k, v in get_type_hints(cls, include_extras=True).items() if get_origin(v) is NotRequired}


def get_required_keys(cls):
    return cls.__required_keys__ - get_explicit_unrequired_keys(cls)


def get_optional_keys(cls):
    return cls.__optional__keys - get_explicit_required_keys(cls)


class SetupDefaults(unittest.TestCase):
    def setUp(self):
        print("Remember to enable/disable justMyCode('\"debugpy.debugJustMyCode\": false,') in the settings")
        env = gym.make("CartPole-v1")

        self._OBSERVATION_SPACE = env.observation_space
        self._ACTION_SPACE = env.action_space

        self._DEFAULT_CONFIG_DICT: Any = MappingProxyType(asdict(SympolCLIArgs()))
        self._DEFAULT_NAMESPACE = SympolCLIArgs()
        self._INPUT_LENGTH = env.observation_space.shape[0]  # pyright: ignore[reportOptionalSubscript]
        self._DEFAULT_INPUT = jnp.arange(self._INPUT_LENGTH * 2).reshape((2, self._INPUT_LENGTH))
        self._DEFAULT_BATCH: dict[str, chex.Array] = MappingProxyType({"obs": self._DEFAULT_INPUT})  # pyright: ignore[reportAttributeAccessIssue]
        self._ENV_SAMPLE = jnp.arange(self._INPUT_LENGTH)
        model_key = jax.random.PRNGKey(self._DEFAULT_CONFIG_DICT["seed"])
        self._RANDOM_KEY, self._ACTOR_KEY, self._CRITIC_KEY = jax.random.split(model_key, 3)
        self._ACTION_DIM: int = self._ACTION_SPACE.n  # type: ignore[attr-defined]
        self._OBS_DIM: int = self._OBSERVATION_SPACE.shape[0]  # pyright: ignore[reportOptionalSubscript]

    def util_test_state_equivalence(
        self,
        state1: TrainState | ActorTrainState | Any,
        state2: TrainState | ActorTrainState | Any,
        msg="",
        *,
        ignore: Collection[str] = (),
    ):
        """Check if two states are equivalent."""
        # Check if the parameters and indices are equal
        if isinstance(ignore, str):
            ignore = {ignore}
        else:
            ignore = set(ignore)

        for attr in ["params", "indices", "grad_accum", "opt_state"]:
            if attr in ignore:
                continue
            with self.subTest(msg=msg, attr=attr):
                val1 = getattr(state1, attr, None)
                val2 = getattr(state2, attr, None)
                self.assertEqual(val1 is not None, val2 is not None, f"Attribute {attr} not found in both states {msg}")
                if val1 is None and val2 is None:
                    continue
                flat1 = tree.flatten(val1)
                flat_params2 = tree.flatten(val2)
                tree.assert_same_structure(flat1, flat_params2)
                for p1, p2 in zip(flat1, flat_params2):
                    npt.assert_array_equal(p1, p2, err_msg=f"Attribute '{attr}' not equal in both states {msg}")

        # Check if the other attributes are equal
        for attr in set(dir(state1) + dir(state2)) - ignore:
            if not attr.startswith("_") and attr not in [
                "params",
                "indices",
                "grad_accum",
                "opt_state",
                "apply_gradients",
                "tx",
                "replace",
            ]:
                val1 = getattr(state1, attr, None)
                val2 = getattr(state2, attr, None)
                self.assertEqual(
                    val1 is not None, val2 is not None, f"Attribute '{attr}' not found in both states {msg}"
                )
                comp = val1 == val2
                if isinstance(comp, bool):
                    self.assertTrue(comp, f"Attribute '{attr}' not equal in both states: {val1}\n!=\n{val2}\n{msg}")
                else:
                    self.assertTrue(
                        comp.all(), f"Attribute '{attr}' not equal in both states: {val1}\n!=\n{val2}\n{msg}"
                    )

        # NOTE: Apply gradients modifies state

    def _create_original_actor_state(self, model):
        return ActorTrainState.create(
            apply_fn=None,
            params=model.init(self._ACTOR_KEY, jnp.array([self._ENV_SAMPLE])),
            tx=optax.chain(
                optax.clip_by_global_norm(self._DEFAULT_NAMESPACE.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(self._DEFAULT_NAMESPACE.learning_rate_actor),
            ),
            grad_accum=jax.tree.map(jnp.zeros_like, model.init(self._ACTOR_KEY, jnp.array([self._ENV_SAMPLE]))),
            indices=None,
        )

    def _create_critic_state(self, critic):
        return TrainState.create(
            apply_fn=None,
            params=critic.init(self._CRITIC_KEY, jnp.array([self._ENV_SAMPLE])),
            tx=optax.chain(
                optax.clip_by_global_norm(self._DEFAULT_NAMESPACE.max_grad_norm),
                # adam or adamW:
                (optax.adamw if self._DEFAULT_NAMESPACE.adamW else optax.adam)(
                    learning_rate=self._DEFAULT_NAMESPACE.learning_rate_critic
                ),
            ),
        )

    def _create_actor_sdt(self):
        model = Actor_SDT(
            self._ACTION_DIM,
            depth=self._DEFAULT_NAMESPACE.depth,
            temperature=self._DEFAULT_NAMESPACE.temperature,
            action_type=self._DEFAULT_NAMESPACE.action_type,
        )
        state = self._create_original_actor_state(model)
        return model, state

    def _create_critic_mlp(self):
        model = Critic_MLP(
            num_layers=self._DEFAULT_NAMESPACE.num_layers,
            neurons_per_layer=self._DEFAULT_NAMESPACE.neurons_per_layer,
        )
        state = self._create_critic_state(model)
        return model, state

    def _create_actor_mlp(self):
        model = Actor_MLP(
            action_dim=self._ACTION_DIM,
            num_layers=self._DEFAULT_NAMESPACE.num_layers,
            neurons_per_layer=self._DEFAULT_NAMESPACE.neurons_per_layer,
        )
        state = self._create_original_actor_state(model)
        return model, state

    def _create_critic_sdt(self):
        model = Critic_SDT(
            depth=self._DEFAULT_NAMESPACE.depth,
            temperature=self._DEFAULT_NAMESPACE.temperature,
        )
        state = self._create_critic_state(model)
        return model, state

    def _create_actor_sympol(self):
        actor = SYMPOL_RL(
            obs_dim=self._INPUT_LENGTH,
            action_dim=self._ACTION_DIM,
            depth=self._DEFAULT_NAMESPACE.depth,
            n_estimators=self._DEFAULT_NAMESPACE.n_estimators,
            action_type=self._DEFAULT_NAMESPACE.action_type,
        )

        def map_nested_fn(fn):
            """Recursively apply `fn` to key-value pairs of a nested dict."""

            def map_fn(nested_dict):
                return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v)) for k, v in nested_dict.items()}

            return map_fn

        args = self._DEFAULT_NAMESPACE
        actor_key = self._ACTOR_KEY

        # original code
        if args.SWA:
            from optax_swag import swag

            if args.adamW:
                actor_state = ActorTrainState.create(
                    apply_fn=None,
                    params=actor.init(actor_key, jnp.array([self._ENV_SAMPLE])),
                    tx=optax.chain(
                        optax.clip_by_global_norm(args.max_grad_norm),
                        optax.multi_transform(
                            {
                                "estimator_weights": optax.inject_hyperparams(optax.adam)(
                                    args.learning_rate_actor_weights
                                ),
                                "split_values": optax.inject_hyperparams(optax.adam)(
                                    args.learning_rate_actor_split_values
                                ),
                                "split_idx_array": optax.inject_hyperparams(optax.adamw)(
                                    args.learning_rate_actor_split_idx_array
                                ),
                                "leaf_array": optax.inject_hyperparams(optax.adamw)(
                                    args.learning_rate_actor_leaf_array
                                ),
                                "log_std": optax.inject_hyperparams(optax.adamw)(args.learning_rate_actor_log_std),
                            },
                            map_nested_fn(lambda k, _: k),
                        ),
                        swag(10, 2),
                    ),
                    grad_accum=jax.tree.map(jnp.zeros_like, actor.init(actor_key, jnp.array([self._ENV_SAMPLE]))),
                    indices=actor.init_indices(actor_key) if args.actor == "sympol" else None,
                )
            else:
                actor_state = ActorTrainState.create(
                    apply_fn=None,
                    params=actor.init(actor_key, jnp.array([self._ENV_SAMPLE])),
                    tx=optax.chain(
                        optax.clip_by_global_norm(args.max_grad_norm),
                        optax.multi_transform(
                            {
                                "estimator_weights": optax.inject_hyperparams(optax.adam)(
                                    args.learning_rate_actor_weights
                                ),
                                "split_values": optax.inject_hyperparams(optax.adam)(
                                    args.learning_rate_actor_split_values
                                ),
                                "split_idx_array": optax.inject_hyperparams(optax.adam)(
                                    args.learning_rate_actor_split_idx_array
                                ),
                                "leaf_array": optax.inject_hyperparams(optax.adam)(args.learning_rate_actor_leaf_array),
                                "log_std": optax.inject_hyperparams(optax.adam)(args.learning_rate_actor_log_std),
                            },
                            map_nested_fn(lambda k, _: k),
                        ),
                        swag(10, 2),
                    ),
                    grad_accum=jax.tree.map(jnp.zeros_like, actor.init(actor_key, jnp.array([self._ENV_SAMPLE]))),
                    indices=actor.init_indices(actor_key) if args.actor == "sympol" else None,
                )
        else:  # noqa
            if args.adamW:
                actor_state: ActorTrainState = ActorTrainState.create(
                    apply_fn=None,
                    params=actor.init(actor_key, jnp.array([self._ENV_SAMPLE])),
                    tx=optax.chain(
                        optax.clip_by_global_norm(args.max_grad_norm),
                        optax.multi_transform(
                            {
                                "estimator_weights": optax.inject_hyperparams(optax.adam)(
                                    args.learning_rate_actor_weights
                                ),
                                "split_values": optax.inject_hyperparams(optax.adam)(
                                    args.learning_rate_actor_split_values
                                ),
                                "split_idx_array": optax.inject_hyperparams(optax.adamw)(
                                    args.learning_rate_actor_split_idx_array
                                ),
                                "leaf_array": optax.inject_hyperparams(optax.adamw)(
                                    args.learning_rate_actor_leaf_array
                                ),
                                "log_std": optax.inject_hyperparams(optax.adamw)(args.learning_rate_actor_log_std),
                            },
                            map_nested_fn(lambda k, _: k),
                        ),
                    ),
                    grad_accum=jax.tree.map(jnp.zeros_like, actor.init(actor_key, jnp.array([self._ENV_SAMPLE]))),
                    indices=actor.init_indices(actor_key) if args.actor == "sympol" else None,
                )
            else:
                actor_state = ActorTrainState.create(
                    apply_fn=None,
                    params=actor.init(actor_key, jnp.array([self._ENV_SAMPLE])),
                    tx=optax.chain(
                        optax.clip_by_global_norm(args.max_grad_norm),
                        optax.multi_transform(
                            {
                                "estimator_weights": optax.inject_hyperparams(optax.adam)(
                                    args.learning_rate_actor_weights
                                ),
                                "split_values": optax.inject_hyperparams(optax.adam)(
                                    args.learning_rate_actor_split_values
                                ),
                                "split_idx_array": optax.inject_hyperparams(optax.adam)(
                                    args.learning_rate_actor_split_idx_array
                                ),
                                "leaf_array": optax.inject_hyperparams(optax.adam)(args.learning_rate_actor_leaf_array),
                                "log_std": optax.inject_hyperparams(optax.adam)(args.learning_rate_actor_log_std),
                            },
                            map_nested_fn(lambda k, _: k),
                        ),
                    ),
                    grad_accum=jax.tree.map(jnp.zeros_like, actor.init(actor_key, jnp.array([self._ENV_SAMPLE]))),
                    indices=actor.init_indices(actor_key) if args.actor == "sympol" else None,
                )
        return actor, actor_state


class DisableBreakpointsForGUI(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if {"-v", "test*.py"} & set(sys.argv):
            print("disable breakpoint")
            mock.patch("builtins.breakpoint").start()
        else:
            print("enable breakpoint")
