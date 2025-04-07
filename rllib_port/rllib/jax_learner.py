from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Sequence, cast

import jax
import jax.numpy as jnp
from ray.rllib.algorithms.ppo.ppo_learner import PPOLearner as RayPPOLearner
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.learner import GeneralAdvantageEstimation
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import Learner
from ray.rllib.core.learner.tf.tf_learner import TfLearner
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.policy.tf_mixins import (
    EntropyCoeffSchedule,
    KLCoeffMixin,
    LearningRateSchedule,
    ValueNetworkMixin,
)
from ray.rllib.utils.typing import (
    EpisodeType,
    LearningRateOrSchedule,
    ParamRef,
    ResultDict,
    ShouldModuleBeUpdatedFn,
    StateDict,
)

from config_types.args_types import CLIArgs
from rllib_port._sample_batch_to_storage import batch_to_storage
from rllib_port.sympol.sympol_model import SympolRLModel
from utils.get_action_and_value import get_action_and_value, get_action_and_value2
from utils.ppo import compute_gae, update_ppo

if TYPE_CHECKING:
    from ray.rllib.utils.typing import (
        ModuleID,
        Optimizer,
        Param,
        ParamDict,
        TensorType,
    )
    from collections.abc import Hashable

    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.algorithms.ppo.ppo import PPOConfig
    from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleSpec
    from ray.rllib.policy.sample_batch import SampleBatch

    from mlp import Critic_MLP
    from rllib_port.sympol.sympol_module import SympolPPOModule
    from sdt import Critic_SDT
    from utils.utils import ActorTrainState, TrainState


__all__ = [
    "JaxLearner",
    "JaxPPOLearner",
]

logger = logging.getLogger(__name__)


class _TfExample(EntropyCoeffSchedule, KLCoeffMixin, LearningRateSchedule, ValueNetworkMixin):
    pass


class _NoTensorConverter(ConnectorV2):
    """A dummy connector to be used instead of a NumpyToTensor connector when a connector is needed"""

    def __call__(
        self,
        *,
        rl_module: RLModule,  # noqa: ARG002
        batch: dict[str, Any],
        **kwargs,  # noqa: ARG002
    ) -> Any:
        return batch


class JaxLearner(Learner):
    framework = "jax"

    def __init__(
        self,
        *,
        config: "AlgorithmConfig | PPOConfig",
        module_spec: Optional[RLModuleSpec | MultiRLModuleSpec] = None,
        module: Optional[RLModule] = None,
    ):
        super().__init__(config=config, module_spec=module_spec, module=module)
        # Should use learner config
        # TODO
        # possible use config["accumulate_grad_batches"]
        self._accumulate_gradients_every: int = config.rl_module_spec.model_config["accumulate_gradients_every"]
        self._accumulate_gradients_every_initial: int = config.rl_module_spec.model_config["accumulate_gradients_every"]

    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: "AlgorithmConfig",
        batch: dict[str, Any],
        fwd_out: dict[str, TensorType],
    ) -> TensorType:
        # HERE critic is evaluated
        # PPOTorchLearner.compute_loss_for_module
        module: SympolPPOModule = self.module[module_id].unwrapped()
        logprob, entropy, value = get_action_and_value2(
            module.states["actor"].params,
            module.states["critic"].params,
            fwd_out,
            action=batch[Columns.ACTIONS],
            action_type=module.model_config["action_type"],
            actor=module.pi.model,
            critic=module.vf.model,
            actor_state_indices=module.states["actor"].indices,
        )

    # calls configure_optimziers_for_module
    # def configure_optimizers(self) -> None:
    #    return super().configure_optimizers()

    def configure_optimizers_for_module(self, module_id: ModuleID, config: "AlgorithmConfig" = None) -> None:
        # MAYBE NOT NEEDED
        module: SympolPPOModule = self._module[module_id]  # type: ignore[assignment]
        # likely do not need these here
        actor_params, critic_params = self.get_parameters(module)  # Re-enable this line
        # Optimizer is set in init_state
        self._states = module.get_state()

        if False:
            self.register_optimizer(
                module_id=module_id,
                optimizer=optimizer,
                params=actor_params,
                lr_or_lr_schedule=config.lr,
            )
            module.states["actor"].tx = optimizer
            self.register_optimizer(
                module_id=module_id,
                optimizer=optimizer,
                params=critic_params,
                lr_or_lr_schedule=config.lr,
            )

    def apply_gradients(self, gradients_dict: ParamDict) -> None:
        logger.warning("get_param_ref called which is not fully implemented")
        # critic
        critic_grads = gradients_dict["critic"]
        self._states["critic"] = self._states["critic"].apply_gradients(grads=critic_grads)

        # actor
        actor_grads = gradients_dict["actor"]
        actor_grad_accum = jax.tree_util.tree_map(lambda x, y: x + y, actor_grads, self._states["actor"].grad_accum)
        actor_state: ActorTrainState = self._states["actor"].apply_gradients(grads=actor_grads)

        def update_fn():
            grads = jax.tree_util.tree_map(lambda x: x / self._accumulate_gradients_every, actor_grad_accum)
            new_state = actor_state.apply_gradients(
                grads=grads,
                grad_accum=jax.tree_util.tree_map(jnp.zeros_like, grads),
            )
            return new_state

        actor_state = jax.lax.cond(
            actor_state.step % self._accumulate_gradients_every == 0,
            lambda _: update_fn(),
            lambda _: actor_state.replace(grad_accum=actor_grad_accum, step=actor_state.step + 1),
            None,
        )
        self._states["actor"] = actor_state

    def get_parameters(self, module: SympolPPOModule | Any) -> tuple[Sequence[Param], Sequence[Param]]:
        return list(module.states["actor"].params), list(module.states["critic"].params)

    def get_param_ref(self, param: Param) -> Hashable:
        logger.warning("get_param_ref called which is not fully implemented")
        return param

    def compute_gradients(self, loss_per_module: dict[ModuleID, Any], **kwargs) -> ParamDict:
        # TODO: Can this be its own function?
        return super().compute_gradients(loss_per_module, **kwargs)

    def _convert_batch_type(self, batch: MultiAgentBatch) -> MultiAgentBatch:
        # TODO: put on device
        logger.warning("_convert_batch_type called which is not fully implemented")
        length = max(len(b) for b in batch.values())
        batch = MultiAgentBatch(batch, env_steps=length)
        return batch

    @staticmethod
    def _get_clip_function():
        logger.warning("_get_clip_function called which is not fully implemented")
        return super(JaxLearner)._get_clip_function()

    @staticmethod
    def _get_global_norm_function() -> Any:
        logger.warning("_get_global_norm_function called which is not fully implemented")
        return super(JaxLearner)._get_global_norm_function()

    def _get_tensor_variable(self, value: Any, dtype: Any = None, trainable: bool = False) -> TensorType:
        logger.warning("_get_tensor_variable called which is not fully implemented")
        if 0:
            TorchLearner._get_tensor_variable(value, dtype, trainable)
            TfLearner._get_tensor_variable(value, dtype, trainable)
        return super()._get_tensor_variable(value, dtype, trainable)

    @staticmethod
    def _get_optimizer_lr(optimizer: Optimizer) -> float:
        logger.warning("_get_optimizer_lr called which is not fully implemented")
        return super(JaxLearner)._get_optimizer_lr(optimizer)

    @staticmethod
    def _set_optimizer_lr(optimizer: Optimizer, lr: float) -> None:
        logger.warning("_set_optimizer_lr called which is not fully implemented")
        super(JaxLearner)._set_optimizer_lr(optimizer, lr)

    def _get_optimizer_state(self, *args, **kwargs):
        logger.warning("_get_optimizer_state called which is not fully implemented")
        return super()._get_optimizer_state(*args, **kwargs)


class JaxPPOLearner(RayPPOLearner, JaxLearner):
    def build(self, **kwargs) -> None:
        super().build(**kwargs)
        self._rng_key = self.config.learner_config_dict["rng_key"]
        if self._learner_connector is not None and (self.config.add_default_connectors_to_learner_pipeline):
            # super().build adds (PPO) AddOneTsToEpisodesAndTruncate, GeneralAdvantageEstimation
            # Remove NumpyToTensorConnector, if present.
            self._learner_connector.remove(name_or_class="NumpyToTensor")

            # At the end of the pipeline (when the batch is already completed), add the
            # GAE connector, which performs a vf forward pass, then computes the GAE
            # computations, and puts the results of this (advantages, value targets)
            # directly back in the batch. This is then the batch used for
            # `forward_train` and `compute_losses`.
            if self.config.learner_config_dict.get("no_numpy_to_tensor_connector", True):
                idx = -1
                for idx, con in enumerate(self._learner_connector.connectors):  # noqa: B007
                    if con.__class__ is GeneralAdvantageEstimation:
                        break
                if idx >= 0:
                    con = cast("GeneralAdvantageEstimation", self._learner_connector.connectors[idx])
                    con._numpy_to_tensor_connector = _NoTensorConverter()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
            # not needed anymore; state passed to self.vf(obs, state=state)
            # self._learner_connector.append(RemoveStateFromBatch())

    def _update(self, batch: dict[str, Any] | SampleBatch, **kwargs) -> tuple[Any, Any, Any]:
        """
        Calls a.o.
        fwd_out = self.module.forward_train(batch)
        loss_per_module = self.compute_losses(fwd_out=fwd_out, batch=batch)
        gradients = self.compute_gradients(loss_per_module)
        postprocessed_gradients = self.postprocess_gradients(gradients)
        self.apply_gradients(postprocessed_gradients)
        """
        # possibly use jit and wrap them all
        TfLearner._untraced_update
        TorchLearner._uncompiled_update
        # get them from somewhere else?
        self.metrics.activate_tensor_mode()
        # fwd_out = self.module.forward_train(batch)
        fwd_out = dict.fromkeys(batch.keys(), None)

        # Use compute_losses for whole module
        if 0:
            loss_per_module = self.compute_losses(fwd_out=fwd_out, batch=batch)
            # calls
            self.compute_loss_for_module
        # However that makes less use of jit:

        loss_per_module = dict.fromkeys(batch.keys(), None)
        if 0:
            fwd_out = {
                mid: self.module._rl_modules[mid]._forward_train(batch[mid], **kwargs)
                for mid in batch.keys()
                if mid in self.module
            }
        for module_id, module_batch in batch.items():
            # Length of the batch entries is minibatch size
            # keys liekely embeddings and "action_dist_inputs"
            # variables:
            module: SympolPPOModule = self.module[module_id]
            actor_state: ActorTrainState = module.states["actor"]
            critic_state: TrainState = self._states["critic"]
            actor = module.pi.model
            critic: Critic_SDT | Critic_MLP = module.vf.model

            args: CLIArgs = CLIArgs(**{k: v for k, v in module.model_config.items() if k in CLIArgs.__annotations__})  # pyright: ignore[reportArgumentType]
            args.n_envs = 1
            # key used to permutate the batch
            self._rng_key, key = jax.random.split(self._rng_key, 2)
            next_obs = module_batch[Columns.OBS]
            next_done = jnp.logical_or(module_batch[Columns.TERMINATEDS], module_batch[Columns.TRUNCATEDS])
            if False:
                storage = batch_to_storage(
                    module_batch,
                    advantages=None and module_batch[Columns.ADVANTAGES],
                    values=None and module_batch[Columns.VF_PREDS],
                    returns=None and module_batch[Columns.VALUE_TARGETS],
                )

                def print_values(msg):
                    return
                    print("Storage", msg, storage.values.shape, ":\n", storage.values[jnp.array([0, -1])])

                print_values("initial")

                for i, (next_obs, next_done) in enumerate(
                    zip(
                        module_batch[Columns.OBS],
                        jnp.logical_or(module_batch[Columns.TERMINATEDS], module_batch[Columns.TRUNCATEDS]),
                    )
                ):
                    storage, action, key = get_action_and_value(
                        actor_state.params,
                        critic_state,
                        next_obs,
                        next_done,
                        storage,
                        i,
                        key,
                        action_type=args.action_type,
                        actor=actor,
                        critic=critic,
                        actor_state_indices=actor_state.indices,
                    )

                # storage, action, key = module.get_action_and_value(next_obs, next_done, key, step=0)

                # TODO: Need values from actions; but value into batch
                # returns = advantages + values
                # rlllib: advantages = module_value_targets - module_vf_preds
                # => returns = advantages + module_vf_preds = module_value_targets

                # NOTE: Alternatively remove the Gae connector and compute critic output and gae here.
                # In original code, next_obs, next_done are the outputs of the n_envs, e.g. length 8

                print_values("after rollout")
                storage = compute_gae(critic_state, next_obs, next_done, storage, critic=critic, args=args)
                print_values("with gae")

            storage2 = batch_to_storage(
                module_batch,
                advantages=module_batch[Columns.ADVANTAGES],
                values=module_batch[Columns.VF_PREDS],
                returns=module_batch[Columns.VALUE_TARGETS],
            )

            # breakpoint()

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
            # print_values("final")
            # Update states
            module.states["actor"] = actor_state
            module.states["critic"] = critic_state
            module.states["module_key"] = key
            loss_per_module[module_id] = loss.mean()
        if 0:
            PPOTorchLearner.compute_loss_for_module
            super().compute_losses
            TorchLearner._update(self, batch, **kwargs)
            fwd_out, loss_per_module, tensor_metrics = TfLearner._update(self, batch, **kwargs)
        return fwd_out, loss_per_module, self.metrics.deactivate_tensor_mode()

    def _update_module_kl_coeff(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        kl_loss: float,
    ) -> None:
        # Note uses:
        config.kl_target
        PPOTorchLearner._update_module_kl_coeff(
            self,  # pyright: ignore[reportArgumentType]
            module_id=module_id,
            config=config,
            kl_loss=kl_loss,
        )
        if False:
            _TfExample.update_kl
            _TfExample._set_kl_coeff
            super()._update_module_kl_coeff(module_id=module_id, config=config, kl_loss=kl_loss)


if TYPE_CHECKING:
    __conf: Any = ...
    JaxLearner(config=__conf)
    JaxPPOLearner(config=__conf)
