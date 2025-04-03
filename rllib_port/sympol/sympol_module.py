from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, TypedDict, Union, cast

import jax
from ray.rllib.algorithms.ppo.default_ppo_rl_module import DefaultPPORLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT
from ray.rllib.core.rl_module.apis import InferenceOnlyAPI

from ray_utilities.jax.distributions.get_distributions_mixin import GetJaxDistributionsMixin
from ray_utilities.jax.jax_module import JaxModule
from rllib_port.mlp.mlp_model import CriticMLPModel
from rllib_port.sympol.sympol_catalog import SympolJaxPPOCatalog
from utils.get_action_and_value import get_action_and_value

if TYPE_CHECKING:
    from utils.utils import ActorTrainState, Storage, TrainState
    import chex
    import gymnasium as gym
    from numpy.typing import NDArray
    from ray.rllib.utils.typing import TensorType

    from config_types.params_types import CLIArgsDict
    from ray_utilities.dummy_encoder import DummyActorCriticEncoder
    from rllib_port.mlp.mlp_model import ActorMLPContinuousModel, ActorMLPModel, CriticMLPModel
    from rllib_port.sdt.sdt_model import ActorSDTModel, CriticSDTModel
    from rllib_port.sympol.sympol_model import SympolRLModel

# for a Intermediate old API to new API Module


class JaxPPOStateDict(TypedDict):
    actor: ActorTrainState
    critic: TrainState
    module_key: int


class SympolPPOModule(GetJaxDistributionsMixin, JaxModule, DefaultPPORLModule):
    # torch code: which should be equivalent
    pi: SympolRLModel | ActorMLPModel | ActorMLPContinuousModel | ActorSDTModel[bool]
    vf: CriticSDTModel | CriticMLPModel
    encoder: DummyActorCriticEncoder
    config: object
    """Deprecated: use model_config instead of config"""

    def __init__(
        self,
        config: int = -1,  # deprecated
        *,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        inference_only: Optional[bool] = None,
        learner_only: bool = False,
        model_config: Union[dict, CLIArgsDict],
        catalog_class=None,
        **kwargs,
    ):
        catalog_class = kwargs.pop("catalog_class", None)
        if catalog_class is None:
            catalog_class = SympolJaxPPOCatalog
        self.model_config: CLIArgsDict
        super().__init__(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=cast("dict", model_config),
            catalog_class=catalog_class,
        )
        # Calls setup

        self.catalog: SympolJaxPPOCatalog
        if self.inference_only:  # and self.framework == "torch":  # modified in setup()
            self.catalog.actor_critic_encoder_config.inference_only = True

        # TorchRLModule init, note DefaultPPORLModule is a subclass of InferenceOnlyAPI
        if self.inference_only and isinstance(self, InferenceOnlyAPI):
            for attr in self.get_non_inference_attributes():
                parts = attr.split(".")
                if not hasattr(self, parts[0]):
                    continue
                target_name = parts[0]
                target_obj = getattr(self, target_name)
                # Traverse from the next part on (if nested).
                for part in parts[1:]:
                    if not hasattr(target_obj, part):
                        target_obj = None
                        break
                    target_name = part
                    target_obj = getattr(target_obj, target_name)
                # Delete, if target is valid.
                if target_obj is not None:
                    delattr(self, target_name)

    def setup(self) -> None:
        super().setup()
        actor = self.pi
        critic = self.vf
        module_key = jax.random.PRNGKey(self.model_config["seed"])
        module_key, actor_key, critic_key = jax.random.split(module_key, 3)

        assert self.observation_space is not None
        sample = self.observation_space.sample()
        actor_state = actor.init_state(actor_key, sample)
        critic_state = critic.init_state(critic_key, sample)

        self.states: JaxPPOStateDict
        self.set_state(
            {
                "actor": actor_state,
                "critic": critic_state,
                "module_key": module_key,
            }
        )

    def to(self, device: Optional[chex.Device] = None):
        # FIXME: Implement proper device handling
        return
        if device is None:
            # put all states on device
            device = jax.local_devices()[0]
        self.states[ACTOR] = jax.device_put(self.states[ACTOR], device)
        self.states[CRITIC] = jax.device_put(self.states[CRITIC], device)

    # region: forward methods

    # Currently Same as DefaultPPOTorchRLModule
    def _forward(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Default forward pass (used for inference and exploration)."""
        output = {}
        # Encoder forward pass.
        encoder_outs = self.encoder(batch)
        # Stateful encoder?
        if Columns.STATE_OUT in encoder_outs:
            output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]
        # Pi head.
        encoder_outs[ENCODER_OUT][ACTOR]
        # print("forward out encoder_outs[ENCODER_OUT][ACTOR])
        encoder_outs[ENCODER_OUT][ACTOR]["state"] = self.states[ACTOR]
        model_out = self.pi(encoder_outs[ENCODER_OUT][ACTOR])
        if self.model_config["action_type"] != "discrete":
            # mean, log_std = model_out
            # TODO: Figure which return values to use
            # if continuous we need a mean and log_std for a non-categorical distribution
            # TODO: Columns.ACTION_LOGP
            output[Columns.ACTION_DIST_INPUTS] = model_out
        else:
            output[Columns.ACTION_DIST_INPUTS] = model_out
        return output

    # Currently Same as DefaultPPOTorchRLModule
    def _forward_train(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Train forward pass (keep embeddings for possible shared value func. call)."""
        output = {}
        encoder_outs = self.encoder(batch)
        output[Columns.EMBEDDINGS] = encoder_outs[ENCODER_OUT][CRITIC]
        if Columns.STATE_OUT in encoder_outs:
            output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]
        encoder_outs[ENCODER_OUT][ACTOR]["state"] = self.states[ACTOR]
        model_out = self.pi(encoder_outs[ENCODER_OUT][ACTOR])
        if self.model_config["action_type"] != "discrete":
            # mean, log_std = model_out
            # TODO: Figure which return values to use
            output[Columns.ACTION_DIST_INPUTS] = model_out
        else:
            output[Columns.ACTION_DIST_INPUTS] = model_out
        return output

    def _forward_inference(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Forward-pass used for action computation without exploration behavior.

        Override this method only, if you need specific behavior for non-exploratory
        action computation behavior. If you have only one generic behavior for all
        phases of training and evaluation, override `self._forward()` instead.

        By default, this calls the generic `self._forward()` method.
        """
        batch = jax.lax.stop_gradient(batch)
        return self._forward(batch, **kwargs)

    def compute_values(
        self,
        batch: dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> TensorType:
        """Computes the value estimates given `batch`.

        Args:
            batch: The batch to compute value function estimates for.
            embeddings: Optional embeddings already computed from the `batch` (by
                another forward pass through the model's encoder (or other subcomponent
                that computes an embedding). For example, the caller of thie method
                should provide `embeddings` - if available - to avoid duplicate passes
                through a shared encoder.

        Returns:
            A tensor of shape (B,) or (B, T) (in case the input `batch` has a
            time dimension.

            Attention:
                That the last value dimension should already be squeezed out (not 1!).
        """
        if embeddings is None:
            # Separate vf-encoder.
            if hasattr(self.encoder, "critic_encoder"):
                batch_ = batch
                if self.is_stateful():
                    # The recurrent encoders expect a `(state_in, h)`  key in the
                    # input dict while the key returned is `(state_in, critic, h)`.
                    batch_ = batch.copy()
                    batch_[Columns.STATE_IN] = batch[Columns.STATE_IN][CRITIC]
                embeddings = self.encoder.critic_encoder(batch_)[ENCODER_OUT]  # pyright: ignore[reportOptionalCall]
            # Shared encoder.
            else:
                embeddings = self.encoder(batch)[ENCODER_OUT][CRITIC]

        # Value head. Should not be a list eve in continous case.
        vf_out = self.vf(embeddings, state=self.states["critic"])  # type: ignore[arg-type]
        vf_out = vf_out.squeeze(-1)
        batch[Columns.VF_PREDS] = (
            vf_out  # NEW: # TODO: rllib does not add this to batch here, why; do during learner update?  # noqa: E501
        )
        return vf_out

    # endregion

    # region non-rllib interface

    def update_state(self, *, actor: Optional[ActorTrainState], critic: Optional[TrainState]):
        """Update the actor and critic states."""
        if actor:
            self.states[ACTOR] = actor
        else:
            self.states[CRITIC] = critic

    def get_action_and_value(
        self,
        next_obs: NDArray,
        next_done: NDArray,
        storage: Storage,
        step: int,
        key: chex.PRNGKey,
    ) -> tuple[Storage, Any | chex.Array, chex.PRNGKey]:
        storage, action, key = get_action_and_value(
            actor_state_params=self.states[ACTOR].params,
            critic_state=self.states[CRITIC],
            next_obs=next_obs,
            next_done=next_done,
            storage=storage,
            step=step,
            key=key,
            action_type=self.model_config["action_type"],
            actor=self.pi.model,
            critic=self.vf.model,
            actor_state_indices=self.states[ACTOR].indices,  # pyright: ignore[reportAttributeAccessIssue]
        )
        return storage, action, key

    def parameters(self) -> tuple[jax.Array, jax.Array]:
        return self.states[ACTOR].params, self.states[CRITIC].params


if TYPE_CHECKING:
    SympolPPOModule(model_config={})
