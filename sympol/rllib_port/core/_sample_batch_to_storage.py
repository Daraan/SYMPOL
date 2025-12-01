from typing import Optional

import jax
import jax.numpy as jnp
from ray.rllib.core.columns import Columns
from ray.rllib.policy.sample_batch import SampleBatch

from sympol.utils.utils import StorageNoValues


def batch_to_storage(
    batch: SampleBatch[jax.Array],
    # values: Optional[jax.Array] = None,
    advantages: Optional[jax.Array] = None,
    returns: Optional[jax.Array] = None,
) -> StorageNoValues:
    """
    Convert a SampleBatch to a Storage object.

    Args:
        batch (SampleBatch): The SampleBatch to convert.

    Returns:
        Storage: The converted Storage object.
    """
    # Convert the batch to a Storage object
    # Original code expects shape (len, n_envs), ray does not report all envs; rehsping to -1
    rewards = batch[Columns.REWARDS].reshape(-1, 1)
    assert len(rewards.shape) == 2
    storage = StorageNoValues(
        # values=jnp.zeros_like(rewards) if values is None else values.reshape(-1, 1),
        advantages=jnp.zeros_like(rewards) if advantages is None else advantages.reshape(-1, 1),
        obs=batch[Columns.OBS][:, None, ...],
        actions=batch[Columns.ACTIONS][:, None, ...],
        dones=batch[SampleBatch.DONES].reshape(-1, 1),
        logprobs=batch[Columns.ACTION_LOGP].reshape(-1, 1),
        returns=batch[Columns.VALUE_TARGETS] if returns is None else returns,
        rewards=rewards,
    )
    return storage
