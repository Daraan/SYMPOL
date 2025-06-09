from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional, cast

import jax
import jax.numpy as jnp
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing
from typing_extensions import TypeAliasType

from ray_utilities.jax.distributions.jax_distributions import RLlibToJaxDistribution
from ray_utilities.jax.math import explained_variance
from utils.get_action_and_value import get_action_and_value2

if TYPE_CHECKING:
    import chex
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.models.distributions import Distribution as RllibDistribution

    from rllib_port.core.sympol_module import SympolPPOModule

    try:
        from ray.rllib.algorithms.ppo.default_ppo_rl_module import DefaultPPORLModule
    except ModuleNotFoundError:
        # Refactoring of ray
        from ray.rllib.algorithms.ppo.ppo_rl_module import (
            PPORLModule as DefaultPPORLModule,  # pyright: ignore[reportPrivateImportUsage]
        )

logger = logging.getLogger(__name__)


_return_signature = tuple[
    jnp.ndarray,  # total loss
    # metrics; TODO: possibly use a dict here
    "tuple[chex.Numeric, chex.Numeric, chex.Numeric, chex.Numeric, chex.Numeric, chex.Numeric]",
]

ComputeLossFunction = TypeAliasType(
    "ComputeLossFunction",
    Callable[
        [dict, dict, float, float, type[RLlibToJaxDistribution], type[RLlibToJaxDistribution]],
        _return_signature,
    ],
)


def make_jax_compute_loss_function(module: SympolPPOModule, config: PPOConfig):
    """
    Note:
        All config an module attributes are treated as constant and should not be changed.

        module needs a compute_values method that is compatible with jax.
    """
    if TYPE_CHECKING:
        jax.jit = lambda func, *args, **kwargs: func  # noqa: ARG005
    action_dist_class_train: type[RLlibToJaxDistribution | RllibDistribution]
    action_dist_class_exploration: type[RLlibToJaxDistribution | RllibDistribution]
    action_dist_class_train = module.get_train_action_dist_cls()
    action_dist_class_exploration = module.get_exploration_action_dist_cls()

    @jax.jit
    def jax_compute_loss_for_module(
        critic_state_params,
        batch: dict,
        fwd_out: dict,
        curr_entropy_coeffs: float,
        curr_kl_coeffs: Optional[float],
    ) -> _return_signature:
        if Columns.LOSS_MASK in batch:  # NOTE: when jitted needs to be always/never present
            mask = batch[Columns.LOSS_MASK]
            num_valid = jnp.sum(mask)

            def possibly_masked_mean(a: jax.Array):
                return jnp.sum(jnp.where(mask, a, 0.0)) / num_valid

        else:
            possibly_masked_mean = jnp.mean

        curr_action_dist = action_dist_class_train.from_logits(fwd_out[Columns.ACTION_DIST_INPUTS])
        # Rllib comment: We should ideally do this in the LearnerConnector
        prev_action_dist = action_dist_class_exploration.from_logits(batch[Columns.ACTION_DIST_INPUTS])

        logp_ratio = jnp.exp(curr_action_dist.logp(batch[Columns.ACTIONS]) - batch[Columns.ACTION_LOGP])

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if config.use_kl_loss:
            action_kl = cast("jnp.ndarray", prev_action_dist.kl(curr_action_dist))
            mean_kl_loss = possibly_masked_mean(action_kl)
        else:
            mean_kl_loss = jnp.zeros((1,))  # device=logp_ratio.device # jax has no device :/

        curr_entropy = cast("jnp.ndarray", curr_action_dist.entropy())
        mean_entropy = possibly_masked_mean(curr_entropy)

        surrogate_loss = jnp.minimum(
            batch[Postprocessing.ADVANTAGES] * logp_ratio,  # XXX make argument
            batch[Postprocessing.ADVANTAGES]
            * jnp.clip(
                logp_ratio,
                1 - config.clip_param,  # pyright: ignore[reportOperatorIssue]
                1 + config.clip_param,  # pyright: ignore[reportOperatorIssue]
            ),
        )

        # Compute a value function loss.
        if config.use_critic:
            value_fn_out = cast(
                "jnp.ndarray",
                module.compute_values(
                    batch, parameters=critic_state_params, embeddings=fwd_out.get(Columns.EMBEDDINGS)
                ),
            )  # XXX jit compatible?
            # Masked values have loss 0
            vf_loss = jnp.square(value_fn_out - batch[Postprocessing.VALUE_TARGETS])
            vf_loss_clipped = jnp.clip(vf_loss, 0, config.vf_clip_param)
            mean_vf_loss = possibly_masked_mean(vf_loss_clipped)
            mean_vf_unclipped_loss = possibly_masked_mean(vf_loss)
        # Ignore the value function -> Set all to 0.0.
        else:
            z = jnp.zeros((1,), device=surrogate_loss.device)
            value_fn_out = mean_vf_unclipped_loss = vf_loss_clipped = mean_vf_loss = z

        total_loss = possibly_masked_mean(
            -surrogate_loss + config.vf_loss_coeff * vf_loss_clipped - (curr_entropy_coeffs * curr_entropy)
        )

        # Add mean_kl_loss (already processed through `possibly_masked_mean`),
        # if necessary.
        if config.use_kl_loss:
            total_loss += curr_kl_coeffs * mean_kl_loss

        # Return the total loss and log values
        policy_loss_key = -possibly_masked_mean(surrogate_loss)
        variance_explained = explained_variance(batch[Postprocessing.VALUE_TARGETS], value_fn_out)
        return total_loss, (
            mean_entropy,
            mean_vf_loss,
            mean_vf_unclipped_loss,
            variance_explained,
            policy_loss_key,
            mean_kl_loss,
        )

    return jax_compute_loss_for_module
