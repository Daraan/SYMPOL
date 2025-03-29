from dataclasses import asdict
import sys
from types import MappingProxyType
from typing import Any, TYPE_CHECKING
from unittest import mock
import unittest

import jax
import jax.numpy as jnp
from typing_extensions import NotRequired, Required, get_origin, get_type_hints

from config_types.args_types import CLIArgs
from mlp import Actor_MLP, Critic_MLP
from sdt import Actor_SDT, Critic_SDT

if TYPE_CHECKING:
    import chex

fixed_args = mock.patch.object(sys, "argv", ["file.py", "--no-render_env"])
clean_args = mock.patch.object(sys, "argv", ["file.py"])
"""Use when comparing to CLIArgs"""


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
        self._DEFAULT_CONFIG_DICT: Any = MappingProxyType(asdict(CLIArgs()))
        self._DEFAULT_NAMESPACE = CLIArgs()
        self._INPUT_LENGTH = 4
        self._DEFAULT_INPUT = jnp.arange(self._INPUT_LENGTH * 2).reshape((2, self._INPUT_LENGTH))
        self._ENV_SAMPLE = jnp.arange(self._INPUT_LENGTH)
        self._RANDOM_KEY: chex.PRNGKey = jax.random.PRNGKey(0)
        self._ACTOR_KEY: chex.PRNGKey = jax.random.PRNGKey(1)
        self._CRITIC_KEY: chex.PRNGKey = jax.random.PRNGKey(2)

    def _create_actor_sdt(self):
        return Actor_SDT(
            action_dim=1,
            depth=self._DEFAULT_NAMESPACE.depth,
            temperature=self._DEFAULT_NAMESPACE.temperature,
            action_type=self._DEFAULT_NAMESPACE.action_type,
        )

    def _create_critic_mlp(self):
        return Critic_MLP(
            num_layers=self._DEFAULT_NAMESPACE.num_layers,
            neurons_per_layer=self._DEFAULT_NAMESPACE.neurons_per_layer,
        )

    def _create_actor_mlp(self):
        return Actor_MLP(
            action_dim=1,
            num_layers=self._DEFAULT_NAMESPACE.num_layers,
            neurons_per_layer=self._DEFAULT_NAMESPACE.neurons_per_layer,
        )

    def _create_critic_sdt(self):
        return Critic_SDT(
            depth=self._DEFAULT_NAMESPACE.depth,
            temperature=self._DEFAULT_NAMESPACE.temperature,
        )


class DisableBreakpointsForGUI(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if {"-v", "test*.py"} & set(sys.argv):
            print("disable breakpoint")
            mock.patch("builtins.breakpoint").start()
        else:
            print("enable breakpoint")
