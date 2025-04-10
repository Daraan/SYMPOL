"""
SampleBatch _concat_values does not support jax.

This monkeypatch allows for concatenation of JAX arrays.
"""

import jax.numpy as jnp
from ray.rllib.policy import sample_batch

_original_concat = sample_batch._concat_values


def supports_jax_concat(*values, time_major=None):
    if isinstance(values[0], jnp.ndarray):
        return jnp.concatenate(values, axis=1 if time_major else 0)
    return _original_concat(*values, time_major=time_major)


# module_advantages are are jax.Array when returned by the GeneralAdvantageEstimation
sample_batch._concat_values = supports_jax_concat
