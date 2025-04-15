from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, cast

import jax
import jax.numpy as jnp
import numpy as np
from ray.rllib.algorithms.ppo.ppo import (
    LEARNER_RESULTS_KL_KEY,
    LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY,
    LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY,
    PPOConfig,
)
from ray.rllib.algorithms.ppo.ppo_learner import PPOLearner as RayPPOLearner
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.learner import GeneralAdvantageEstimation
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import ENTROPY_KEY, POLICY_LOSS_KEY, VF_LOSS_KEY, Learner
from ray.rllib.core.learner.tf.tf_learner import TfLearner
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
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
from utils.ppo import compute_gae, update_ppo

from ._jax_compute_loss_for_module import make_jax_compute_loss_function
from ._sample_batch_to_storage import batch_to_storage

if TYPE_CHECKING:
    from collections.abc import Hashable

    import chex
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.rllib.algorithms.ppo.ppo import PPOConfig
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleSpec
    from ray.rllib.utils.typing import (
        ModuleID,
        Optimizer,
        Param,
        ParamDict,
        TensorType,
    )

    from mlp import Critic_MLP
    from ray_utilities.typing.jax import type_grad_and_value
    from rllib_port.core.sympol_module import JaxPPOStateDict, SympolPPOModule
    from sdt import Critic_SDT
    from utils.utils import ActorTrainState, TrainState

    jax.jit = lambda func, *args, **kwargs: func  # noqa: ARG005


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


class _LimitedToNumpyConverter(ConnectorV2):
    """
    Converts Jax arrays to numpy to pass trough the SampleBatch converter

    Experimental might slow down the training;
    jax -> numpy -> jax

    Used for GAE results
    """

    def __call__(
        self,
        *,
        rl_module: RLModule | MultiRLModule,  # noqa: ARG002
        batch: dict[str, Any],
        **kwargs,  # noqa: ARG002
    ) -> Any:
        # Code from NumpyToTensor

        is_single_agent = False
        is_multi_rl_module = isinstance(rl_module, MultiRLModule)
        # `data` already a ModuleID to batch mapping format.
        if not (is_multi_rl_module and all(c in rl_module._rl_modules for c in batch)):  # pyright: ignore[reportAttributeAccessIssue]
            is_single_agent = True
            batch = {DEFAULT_MODULE_ID: batch}

        for module_id, module_data in batch.copy().items():
            infos = module_data.pop(Columns.INFOS, None)
            for k in ("advantages",):
                module_data[k] = np.asarray(module_data[k])
            if infos is not None:
                module_data[Columns.INFOS] = infos
            # Early out with data under(!) `DEFAULT_MODULE_ID`, b/c we are in plain
            # single-agent mode.
            if is_single_agent:
                return module_data
            batch[module_id] = module_data

        return batch
        # batch[Columns.ADVANTAGES] = jax.device_get(batch[Columns.ADVANTAGES])
        # batch["value_targets"] = jax.device_get(batch["value_targets"])


class JaxLearner(Learner):
    framework = "jax"

    def __init__(
        self,
        *,
        config: "AlgorithmConfig | PPOConfig",
        module_spec: Optional[RLModuleSpec | MultiRLModuleSpec] = None,
        module: Optional[RLModule] = None,
    ):
        # calls configure_optimziers_for_module
        super().__init__(config=config, module_spec=module_spec, module=module)
        # Should use learner config
        # TODO
        # possible use config["accumulate_grad_batches"]
        self._accumulate_gradients_every: int = config.learner_config_dict["accumulate_gradients_every"]
        self._accumulate_gradients_every_initial: int = config.learner_config_dict["accumulate_gradients_every"]
        # XXX Possibly do not keep them and access via self.module!
        self._states: dict[ModuleID, JaxPPOStateDict] = {}

    # calls configure_optimziers_for_module
    # def configure_optimizers(self) -> None:
    #    return super().configure_optimizers()

    def configure_optimizers_for_module(self, module_id: ModuleID, config: "AlgorithmConfig") -> None:
        # MAYBE NOT NEEDED
        module: SympolPPOModule = self._module[module_id]  # type: ignore[assignment]
        # likely do not need these here
        actor_params, critic_params = self.get_parameters(module)  # Re-enable this line
        # Optimizer is set in init_state
        self._states[module_id] = module.get_state(inference_only=False)

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

    # jittable
    def apply_gradients(
        self,
        # Normally dict[Hashable | ParamRef, Param]
        gradients_dict: dict[ModuleID, dict[Literal["actor", "critic"], Any]],
        *,
        states: dict[ModuleID, JaxPPOStateDict],
    ) -> dict[ModuleID, JaxPPOStateDict]:
        for module_id in self.module.keys():
            module_grads = gradients_dict[module_id]
            critic_grads = module_grads["critic"]
            states[module_id]["critic"] = states[module_id]["critic"].apply_gradients(grads=critic_grads)

            # actor
            actor_grads = module_grads["actor"]
            actor_grad_accum = jax.tree_util.tree_map(
                lambda x, y: x + y, actor_grads, states[module_id]["actor"].grad_accum
            )
            actor_state: ActorTrainState = states[module_id]["actor"].apply_gradients(grads=actor_grads)

            def update_fn(actor_state=actor_state, actor_grad_accum=actor_grad_accum):
                grads = jax.tree_util.tree_map(lambda x: x / self._accumulate_gradients_every, actor_grad_accum)
                new_state = actor_state.apply_gradients(
                    grads=grads,
                    grad_accum=jax.tree_util.tree_map(jnp.zeros_like, grads),
                )
                return new_state

            actor_state = jax.lax.cond(
                actor_state.step % self._accumulate_gradients_every == 0,
                lambda _: update_fn(),
                lambda _, actor_state=actor_state, actor_grad_accum=actor_grad_accum: actor_state.replace(
                    grad_accum=actor_grad_accum, step=actor_state.step + 1
                ),
                None,
            )
            states[module_id]["actor"] = actor_state
        return states

    def get_parameters(self, module: SympolPPOModule | Any) -> tuple[Sequence[Param], Sequence[Param]]:
        logger.warning("JaxLearner.get_parameters called which is not fully implemented", stacklevel=2)
        return list(module.states["actor"].params), list(module.states["critic"].params)

    def get_param_ref(self, param: Param) -> Hashable:
        # Reference to param: self._params[param_ref] = param
        logger.warning("JaxLearner.get_param_ref called which is not fully implemented")
        return param

    def compute_gradients(self, *args, **kwargs) -> ParamDict:  # noqa: ARG002
        # TODO: Can this be its own function?
        logger.warning("compute_gradients called which is not used in the jax implementation")
        raise NotImplementedError("compute_gradients not implemented for jax")

    def _convert_batch_type(self, batch: MultiAgentBatch) -> MultiAgentBatch:
        # TODO: put on device
        logger.warning("_convert_batch_type called which is not fully implemented")
        length = max(len(b) for b in batch.values())
        batch = MultiAgentBatch(batch, env_steps=length)
        return batch

    @staticmethod
    def _get_clip_function():
        logger.warning("_get_clip_function called which is not fully implemented")
        if 0:
            # returns
            from ray.rllib.utils.tf_utils import clip_gradients
            from ray.rllib.utils.torch_utils import clip_gradients
        # possibly use optax.clip; but needs to be in transformation pipeline
        from ray_utilities.jax.math import clip_gradient

        # Has wrong interface
        return clip_gradient

    @staticmethod
    def _get_global_norm_function() -> Any:
        logger.warning("_get_global_norm_function called which is not fully implemented")
        return super(JaxLearner)._get_global_norm_function()

    def _get_tensor_variable(self, value: Any, dtype: Any = None, trainable: bool = False) -> TensorType:
        # TODO: is kl_coeffs a variable that is learned?
        logger.warning("_get_tensor_variable called which is not fully implemented", stacklevel=2)
        if 0:
            TorchLearner._get_tensor_variable(value, dtype, trainable)
            TfLearner._get_tensor_variable(value, dtype, trainable)
        v = jnp.array(value, dtype=dtype)
        if not trainable:
            v = jax.lax.stop_gradient(v)
        return v

    @staticmethod
    def _get_optimizer_lr(optimizer: Optimizer) -> float:
        logger.warning("_get_optimizer_lr called which is not fully implemented")
        return super(JaxLearner)._get_optimizer_lr(optimizer)

    @staticmethod
    def _set_optimizer_lr(optimizer: Optimizer, lr: float) -> None:
        logger.warning("_set_optimizer_lr called which is not fully implemented")
        # Needs to change opt_state
        # TODO: reduce lr not implemented
        super(JaxLearner)._set_optimizer_lr(optimizer, lr)

    def _get_optimizer_state(self, *args, **kwargs):
        logger.warning("_get_optimizer_state called which is not fully implemented")
        return super()._get_optimizer_state(*args, **kwargs)


class JaxPPOLearner(RayPPOLearner, JaxLearner):
    def build(self, **kwargs) -> None:
        super().build(**kwargs)
        self._legacy = self.config.learner_config_dict.get("legacy", True)
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
                    # TODO: if using no converter need monkeypatch; maybe convert there
                    # con._numpy_to_tensor_connector = _LimitedToNumpyConverter()  # pyright: ignore[reportPrivateUsage]
            # not needed anymore; state passed to self.vf(obs, state=state)
            # self._learner_connector.append(RemoveStateFromBatch())
        self._compute_loss_for_modules = {
            module_id: make_jax_compute_loss_function(
                module,  # pyright: ignore[reportArgumentType]
                self.config,  # pyright: ignore[reportArgumentType]
            )
            for module_id, module in self.module.items()
        }
        if TYPE_CHECKING:
            self._forward_with_grads = type_grad_and_value(self._jax_forward_pass)
        else:
            self._forward_with_grads = jax.jit(jax.value_and_grad(self._jax_forward_pass, has_aux=True, argnums=(0,)))
        self._update_jax = jax.jit(self._update_jax)

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
            args: CLIArgs = CLIArgs(
                **{  # pyright: ignore[reportArgumentType]
                    k: v for k, v in module.model_config.items() if k in CLIArgs.__annotations__
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

    @staticmethod
    def _get_state_parameters(
        states: dict[ModuleID, JaxPPOStateDict],
    ) -> dict[ModuleID, dict[Literal["actor", "critic"], Any]]:
        parameters: dict[ModuleID, dict[Literal["actor", "critic"], Any]] = dict.fromkeys(
            states.keys(), cast("dict", None)
        )
        for module_id, state in states.items():
            parameters[module_id] = {
                "actor": state["actor"].params,
                "critic": state["critic"].params,
            }
        return parameters

    def compute_loss_for_module(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        critic_state_params: Optional[jax.Array],
        module_id: ModuleID,
        config: "AlgorithmConfig | PPOConfig",  # noqa: ARG002
        batch: SampleBatch | dict[str, Any],
        fwd_out: dict[str, TensorType],
        curr_entropy_coeff: float,
        curr_kl_coeff: Optional[float],
    ) -> tuple[TensorType, dict[str, chex.Numeric]]:
        # jittable and grad wrt critic_state_params
        # TODO: Why is there a Loss Mask?
        (
            total_loss,
            (
                mean_entropy,
                mean_vf_loss,
                mean_vf_unclipped_loss,
                variance_explained,
                policy_loss_key,
                mean_kl_loss,
            ),
        ) = self._compute_loss_for_modules[module_id](
            # batch is a SampleBatch which is not compatible
            critic_state_params if critic_state_params is not None else self._states[module_id]["critic"].params,
            batch=batch,
            fwd_out=fwd_out,
            curr_entropy_coeffs=curr_entropy_coeff,
            curr_kl_coeffs=curr_kl_coeff,
        )

        # Return the total loss.
        return total_loss, {
            POLICY_LOSS_KEY: policy_loss_key,
            VF_LOSS_KEY: mean_vf_loss,
            LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY: mean_vf_unclipped_loss,
            LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY: variance_explained,
            ENTROPY_KEY: mean_entropy,
            LEARNER_RESULTS_KL_KEY: mean_kl_loss,
        }

    def compute_losses(self, *, fwd_out: ResultDict[str, Any], batch: ResultDict[str, Any]):
        """
        NOTE:
            Use _jax_compute_losses instead of this function to compute gradients
        """
        logger.warning("compute_losses called, which makes no use of jit - additional step in shedule")
        curr_entropy_coeffs, curr_kl_coeffs = self._generate_curr_coeffs()
        loss_per_module, aux_data = self._jax_compute_losses(
            parameters=self._get_state_parameters(self._states),
            fwd_out=fwd_out,
            batch=batch,
            curr_entropy_coeffs=curr_entropy_coeffs,
            curr_kl_coeffs=curr_kl_coeffs,
        )
        return loss_per_module

    def _jax_compute_losses(
        self,
        parameters: dict[ModuleID, dict[Literal["actor", "critic"], jax.Array]],
        fwd_out: dict[str, Any],
        batch: dict[str, Any],
        curr_entropy_coeffs: dict[ModuleID, float],
        curr_kl_coeffs: Optional[dict[ModuleID, float]] = None,
    ):
        loss_per_module = {}
        aux_data = {}
        from ray.rllib.core.rl_module.apis import SelfSupervisedLossAPI

        for module_id, module_state in parameters.items():
            module_batch = batch[module_id]
            module_fwd_out = fwd_out[module_id]
            module = self.module[module_id].unwrapped()
            if isinstance(module, SelfSupervisedLossAPI):
                logger.error("Self-supervised loss not implemented with jax suport")
                loss = module.compute_self_supervised_loss(
                    learner=self,
                    module_id=module_id,
                    config=self.config.get_config_for_module(module_id),
                    batch=module_batch,
                    fwd_out=module_fwd_out,
                )
            else:
                loss, aux = self.compute_loss_for_module(
                    module_id=module_id,
                    config=self.config.get_config_for_module(module_id),  # pyright: ignore[reportArgumentType]
                    batch=dict(module_batch),
                    fwd_out=module_fwd_out,
                    critic_state_params=module_state["critic"],
                    curr_entropy_coeff=curr_entropy_coeffs[module_id],
                    curr_kl_coeff=curr_kl_coeffs[module_id] if curr_kl_coeffs else None,
                )
                aux_data[module_id] = aux
            loss_per_module[module_id] = loss

        return loss_per_module, aux_data

    def _forward_train_call(
        self, batch, parameters: dict[ModuleID, dict[Literal["actor", "critic"], jax.Array]], **kwargs
    ):
        """jittable"""
        fwd_out = {
            mid: cast("SympolPPOModule", self.module._rl_modules[mid])._forward_train(
                batch[mid], parameters=parameters[mid]["actor"], **kwargs
            )
            for mid in batch.keys()
            if mid in self.module
        }
        return fwd_out

    # NOTE: do not pass indices as states
    # @jax.jit
    # @partial(jax.value_and_grad, has_aux=True, argnums=(0,))
    def _jax_forward_pass(
        self,
        parameters: dict[ModuleID, dict[Literal["actor", "critic"], jax.Array]],
        batch: dict[str, Any],
        curr_entropy_coeffs: dict[ModuleID, float],
        curr_kl_coeffs: Optional[dict[ModuleID, float]] = None,
    ) -> tuple[chex.Numeric, tuple[Any, dict[ModuleID, chex.Numeric], dict[str, Any]]]:
        """
        Note:
            do not use directly use _forward_with_grads
        """
        fwd_out = self._forward_train_call(batch, parameters=parameters)
        loss_per_module, compute_loss_aux = self._jax_compute_losses(
            parameters, fwd_out, batch, curr_entropy_coeffs, curr_kl_coeffs
        )
        # gradient needs a scalar loss:
        return jax.tree.reduce(jnp.sum, loss_per_module), (fwd_out, loss_per_module, compute_loss_aux)

    def _update_jax(
        self,
        states: dict[ModuleID, JaxPPOStateDict],
        batch: dict[str, Any],
        curr_entropy_coeffs: dict[ModuleID, float],
        curr_kl_coeffs: Optional[dict[ModuleID, float]] = None,
    ) -> tuple[dict[ModuleID, JaxPPOStateDict], tuple[Any, dict[ModuleID, chex.Numeric], dict[str, Any]]]:
        parameters = self._get_state_parameters(states)
        gradients: dict[ModuleID, dict[Literal["actor", "critic"], Any]]
        (_all_losses_combined, (fwd_out, loss_per_module_do_not_use, compute_loss_aux)), (gradients,) = (
            self._forward_with_grads(parameters, batch, curr_entropy_coeffs, curr_kl_coeffs)  # pyright: ignore[reportArgumentType]
        )
        if 0:
            # consider if implementation is necessary
            self.postprocess_gradients_for_module
            postprocessed_gradients: dict = self.postprocess_gradients(gradients)
        else:
            postprocessed_gradients = gradients
        new_states = self.apply_gradients(postprocessed_gradients, states=states)
        return new_states, (fwd_out, loss_per_module_do_not_use, compute_loss_aux)

    def _generate_curr_coeffs(self):
        curr_entropy_coeffs = {}
        curr_kl_coeffs = {}
        for module_id in self.module.keys():
            curr_entropy_coeffs[module_id] = self.entropy_coeff_schedulers_per_module[module_id].get_current_value()
            if self.config.get_config_for_module(module_id):  # TODO: This s
                curr_kl_coeffs[module_id] = self.curr_kl_coeffs_per_module[module_id]
            else:
                curr_kl_coeffs[module_id] = 0.0
        return curr_entropy_coeffs, curr_kl_coeffs

    def _update(self, batch: dict[str, Any] | SampleBatch, **kwargs) -> tuple[Any, Any, Any]:
        """
        NOTE: The amount of processed data is minibatch_size * epochs

        Calls a.o.
        fwd_out = self.module.forward_train(batch)
        loss_per_module = self.compute_losses(fwd_out=fwd_out, batch=batch)
        gradients = self.compute_gradients(loss_per_module)
        postprocessed_gradients = self.postprocess_gradients(gradients)
        self.apply_gradients(postprocessed_gradients)
        """
        # possibly use jit and wrap them all
        if 0:
            TfLearner._untraced_update
            TorchLearner._uncompiled_update
        # get them from somewhere else?
        self.metrics.activate_tensor_mode()
        # fwd_out = self.module.forward_train(batch)
        # Cannot pass SampleBatch as input
        if self._legacy:
            # NOTE: Performs PPO update already; updates states in place
            # Does NOT fill fwd_out
            fwd_out, loss_per_module = self._legacy_update(batch, **kwargs)
        else:
            # TODO: fwd_out["default_policy"]["embeddings"] has many keys
            curr_entropy_coeffs, curr_kl_coeffs = self._generate_curr_coeffs()
            new_states, (fwd_out, loss_per_module, compute_loss_aux) = self._update_jax(
                states=self._states,
                batch={mid: dict(v) for mid, v in batch.items()},
                curr_entropy_coeffs=curr_entropy_coeffs,
                curr_kl_coeffs=curr_kl_coeffs,
            )
            self._states = new_states
            self.module.set_state(self._states)

            # Log important loss stats.
            for module_id in fwd_out.keys():
                self.metrics.log_dict(
                    compute_loss_aux[module_id],
                    key=module_id,
                    window=1,  # <- single items (should not be mean/ema-reduced over time).
                )
        return fwd_out, loss_per_module, self.metrics.deactivate_tensor_mode()

    def _update_module_kl_coeff(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        kl_loss: float,
    ) -> None:
        # Note uses:
        # TODO: check if this is called and should it be called
        logger.warning("_update_module_kl_coeff called which is only minimally implemented")
        # Does not use any torch functions; uses kl_target and curr_kl_coeffs_per_module
        PPOTorchLearner._update_module_kl_coeff(
            self,  # pyright: ignore[reportArgumentType]
            module_id=module_id,
            config=config,
            kl_loss=kl_loss,
        )


# pyright: reportAbstractUsage=information
if TYPE_CHECKING:
    __conf: Any = ...
    JaxLearner(config=__conf)
    JaxPPOLearner(config=__conf)
