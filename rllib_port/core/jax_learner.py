from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from ray.rllib.core.columns import Columns
from ray.rllib.policy.tf_mixins import (
    EntropyCoeffSchedule,
    KLCoeffMixin,
    LearningRateSchedule,
    ValueNetworkMixin,
)

from config_types.args_types import SympolCLIArgs
from ray_utilities.jax.ppo.jax_ppo_learner import JaxPPOLearner
from utils.ppo import update_ppo

from ._sample_batch_to_storage import batch_to_storage

if TYPE_CHECKING:
    from ray.rllib.policy.sample_batch import SampleBatch

    from mlp import Critic_MLP
    from rllib_port.core.sympol_module import SympolPPOModule
    from sdt import Critic_SDT
    from utils.utils import ActorTrainState, TrainState

__all__ = [
    "JaxPPOLearnerWithLegacy",
]

logger = logging.getLogger(__name__)


class _TfExample(EntropyCoeffSchedule, KLCoeffMixin, LearningRateSchedule, ValueNetworkMixin):
    pass

    # batch[Columns.ADVANTAGES] = jax.device_get(batch[Columns.ADVANTAGES])
    # batch["value_targets"] = jax.device_get(batch["value_targets"])


class JaxPPOLearnerWithLegacy(JaxPPOLearner):
    def build(self, **kwargs) -> None:
        super().build(**kwargs)
        self._legacy = self.config.learner_config_dict.get("legacy", True)

    def _update(self, batch: dict[str, Any] | SampleBatch, **kwargs) -> tuple[Any, Any, Any]:
        if self._legacy:
            self.metrics.activate_tensor_mode()
            # NOTE: Performs PPO update already; updates states in place
            # Does NOT fill fwd_out
            fwd_out, loss_per_module = self._legacy_update(batch, **kwargs)
            return fwd_out, loss_per_module, self.metrics.deactivate_tensor_mode()
        return super()._update(batch, **kwargs)

    def _legacy_update(self, batch: dict[str, Any] | SampleBatch) -> tuple[Any, Any]:
        fwd_out = dict.fromkeys(batch.keys(), None)
        loss_per_module = dict.fromkeys(batch.keys(), None)
        for module_id, module_batch in batch.items():
            # Length of the batch entries is minibatch size
            # keys liekely embeddings and "action_dist_inputs"
            # variables:
            module: SympolPPOModule = self.module[module_id]  # pyright: ignore[reportAssignmentType]
            actor_state: ActorTrainState = module.states["actor"]
            critic_state: TrainState = module.states["critic"]
            actor = module.pi.model
            critic: Critic_SDT | Critic_MLP = module.vf.model

            # Need a NameSpace and not a dict
            args: SympolCLIArgs = SympolCLIArgs(
                **{  # pyright: ignore[reportArgumentType]
                    k: v for k, v in module.model_config.items() if k in SympolCLIArgs.__annotations__
                }
            )
            args.n_envs = 1
            # key used to permutate the batch
            self._rng_key, key = jax.random.split(self._rng_key, 2)
            if False:
                next_obs = module_batch[Columns.OBS]
                next_done = jnp.logical_or(module_batch[Columns.TERMINATEDS], module_batch[Columns.TRUNCATEDS])
                # full legacy use compute_action_and_value + gae

            storage2 = batch_to_storage(
                module_batch,
                advantages=module_batch[Columns.ADVANTAGES],
                # TODO: could get GAE via jax here. - test speed
                # values=module_batch.get(Columns.VF_PREDS, None),  # <- not needed if we have gae
                returns=module_batch[Columns.VALUE_TARGETS],
            )

            actor_state, critic_state, loss, policy_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
                actor_state,
                critic_state,
                storage2,
                key,
                self._accumulate_gradients_every,
                minibatch_size=self.config.learner_config_dict["legacy_minibatch_size"],
                n_update_epochs=args.n_update_epochs,
                args=args,
                actor=actor,
                critic=critic,
                actor_state_indices=actor_state.indices,
            )
            # Keeps dict in sync, TODO: should not rely on this and use module.set_state
            module.states["actor"] = actor_state
            module.states["critic"] = critic_state
            module.states["module_key"] = key
            loss_per_module[module_id] = loss.mean()
        return fwd_out, loss_per_module


# pyright: reportAbstractUsage=information
if TYPE_CHECKING:
    __conf: Any = ...
    JaxPPOLearnerWithLegacy(config=__conf)
