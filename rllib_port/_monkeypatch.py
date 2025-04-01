"""
SampleBatch _concat_values does not support jax.

This monkeypatch allows for concatenation of JAX arrays.
"""

from typing import TYPE_CHECKING
from ray.rllib.policy import sample_batch
from ray.rllib.utils.framework import try_import_tf, try_import_torch

import jax.numpy as jnp

if TYPE_CHECKING:
    import torch
    import tensorflow as tf
else:
    tf1, tf, tfv = try_import_tf()
    torch, _ = try_import_torch()

_original_concat = sample_batch._concat_values


def supports_jax_concat(*values, time_major=None):
    if isinstance(values[0], jnp.ndarray):
        return jnp.concatenate(values, axis=1 if time_major else 0)
    return _original_concat(*values, time_major=time_major)


sample_batch._concat_values = supports_jax_concat
