from __future__ import annotations

from typing import Any, Generic, Literal, overload

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import normal

from utils import _is_discreteT
from utils.jax_math import entmax15JAX

__all__ = [
    "Actor_SDT",
    "Critic_SDT",
]


def temperature_sigmoid(x, temperature=1.0):
    """Custom sigmoid function with temperature."""
    return 1 / (1 + jnp.exp(-x / temperature))


def entmoid15(x, temperature=1.0):
    return entmax15JAX(jnp.stack([x / temperature, jnp.zeros(x.shape)], axis=-1), axis=-1)[..., 0]


class SubtractiveEntmaxDense(nn.Module):
    features: int  # Number of output features
    temperature: float

    @nn.compact
    def __call__(self, inputs, max_path=False):
        # Create weight and bias variables
        weight = self.param("kernel", normal(0.1), (inputs.shape[-1], self.features))
        bias = self.param("bias", normal(1.0), (self.features,))

        # Compute the dense layer output with bias subtraction
        weight = jax.lax.cond(max_path, lambda x: weight, lambda x: entmax15JAX(weight.T / self.temperature).T, weight)

        output = jnp.dot(inputs, weight) - bias
        output = jax.lax.cond(
            max_path,
            # lambda x: jnp.round(entmoid15(output, self.temperature)),
            lambda x: entmoid15(output, 0.0001),
            lambda x: entmoid15(output, self.temperature),
            output,
        )

        return output


class SDT(nn.Module, Generic[_is_discreteT]):
    input_dim: int
    output_dim: int
    depth: int = 3
    temperature: float = 1.0
    action_type: str = "discrete"  # "continuous"

    def setup(self):
        self.internal_node_num = 2**self.depth - 1
        self.leaf_node_num = 2**self.depth

        self.inner_nodes = nn.Sequential(
            [
                SubtractiveEntmaxDense(
                    self.internal_node_num, self.temperature
                ),  # , use_bias=True, kernel_init=normal(0.1), bias_init=normal(1.0)),
                # nn.sigmoid
                # lambda x: temperature_sigmoid(x, self.temperature)
                # lambda x: entmoid15(x, self.temperature)
                # entmoid15
            ]
        )

        if self.action_type == "discrete":
            self.leaf_nodes = nn.Dense(self.output_dim, use_bias=False, kernel_init=normal(0.1))
            self.stds = None  # nn.Dense(self.output_dim, use_bias=False, kernel_init=normal(0.1))
        else:
            self.leaf_nodes = nn.Dense(self.output_dim, use_bias=False, kernel_init=normal(0.1))
            self.log_std = self.param("log_std", nn.initializers.zeros, (self.output_dim,))

    @overload
    def __call__(self: "SDT[Literal[False]]", x, max_path: bool) -> jax.Array: ...

    @overload
    def __call__(self: "SDT[Literal[True]]", x, max_path: bool) -> list[jax.Array]: ...

    @overload
    def __call__(self: "SDT[Any]", x, max_path: bool) -> jax.Array | list[jax.Array]: ...

    def __call__(self, x, max_path) -> jax.Array | list[jax.Array] | Any:
        batch_size = x.shape[0]
        # x = self._data_augment(x)
        # inner_nodes = self.inner_nodes
        # inner_nodes = entmax15JAX(inner_nodes)

        path_prob = self.inner_nodes(x, max_path=max_path)
        # entmax15JAX(self.inner_nodes(

        path_prob = jnp.expand_dims(path_prob, axis=2)
        path_prob = jnp.concatenate((path_prob, 1 - path_prob), axis=2)

        mu = jnp.ones((batch_size, 1, 1))

        begin_idx = 0
        end_idx = 1

        for layer_idx in range(self.depth):
            path_prob_layer = path_prob[:, begin_idx:end_idx, :]

            mu = jnp.reshape(mu, (batch_size, -1, 1))
            mu = jnp.tile(mu, (1, 1, 2))

            mu = mu * path_prob_layer

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        mu = mu.reshape(batch_size, self.leaf_node_num)

        if self.action_type == "discrete":
            y_pred = self.leaf_nodes(mu)
        else:
            mean = self.leaf_nodes(mu)
            y_pred = [mean, self.log_std]

        return y_pred


class Critic_SDT(nn.Module):
    depth: int = 5
    temperature: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, max_path=False, **kwargs):
        sdt: SDT[Literal[False]] = SDT(
            input_dim=x.shape[-1], output_dim=1, depth=self.depth, temperature=self.temperature
        )
        return sdt(x, False)  # , **kwargs)


class Actor_SDT(nn.Module, Generic[_is_discreteT]):
    action_dim: int
    depth: int = 5
    temperature: float = 1.0
    action_type: Literal["discrete", "continuous"] = "discrete"  # "continuous"

    @overload
    def __call__(
        self: "Actor_SDT[Literal[False]]", obs: jnp.ndarray, max_path: bool = False, **kwargs
    ) -> jnp.ndarray: ...

    @overload
    def __call__(
        self: "Actor_SDT[Literal[True]]", obs: jnp.ndarray, max_path: bool = False, **kwargs
    ) -> list[jnp.ndarray]: ...

    @overload
    def __call__(self, obs: jnp.ndarray, max_path: bool = False, **kwargs) -> jnp.ndarray | list[jnp.ndarray]: ...

    @nn.compact
    def __call__(self, obs: jnp.ndarray, max_path=False, **kwargs):
        sdt: SDT[_is_discreteT] = SDT(
            input_dim=obs.shape[-1],
            output_dim=self.action_dim,
            depth=self.depth,
            temperature=self.temperature,
            action_type=self.action_type,
        )
        return sdt(obs, max_path)  # , **kwargs)
