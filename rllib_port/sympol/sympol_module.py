from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union, cast

import jax
from ray.rllib.algorithms.ppo.default_ppo_rl_module import DefaultPPORLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT
from ray.rllib.core.rl_module.apis import InferenceOnlyAPI

from ray_utilities.jax.jax_model import JaxRLModel
from rllib_port.mlp.mlp_model import CriticMLPModel
from rllib_port.sympol.sympol_catalog import SympolJaxPPOCatalog

if TYPE_CHECKING:
    import gymnasium as gym
    from ray.rllib.utils.typing import TensorType

    from config_types.params_types import SympolCatalogOptions
    from ray_utilities.dummy_encoder import DummyActorCriticEncoder
    from rllib_port.mlp.mlp_model import ActorMLPContinuousModel, ActorMLPModel, CriticMLPModel
    from rllib_port.sdt.sdt_model import ActorSDTModel, CriticSDTModel
    from rllib_port.sympol.sympol_model import SympolRLModel

# for a Intermediate old API to new API Module


class SympolPPOModule(DefaultPPORLModule):
    # torch code: which should be equivalent
    vf: CriticSDTModel | CriticMLPModel
    pi: SympolRLModel | ActorMLPModel | ActorMLPContinuousModel | ActorSDTModel[bool]
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
        model_config: Union[dict, SympolCatalogOptions],
        catalog_class=None,
        **kwargs,
    ):
        catalog_class = kwargs.pop("catalog_class", None)
        if catalog_class is None:
            catalog_class = SympolJaxPPOCatalog
        self.model_config: SympolCatalogOptions
        super().__init__(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=cast(dict, model_config),
            catalog_class=catalog_class,
        )

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

    def setup(self):
        super().setup()
        actor = self.pi
        actory_key = jax.random.PRNGKey(0)  # XXX

        assert self.observation_space is not None
        sample = self.observation_space.sample()
        breakpoint()
        if isinstance(actor, JaxRLModel):
            actor_state = actor.init_state(
                actory_key,
                sample=sample,
            )
        else:
            params = actor.init(actory_key, sample)

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
        model_out = self.pi(encoder_outs[ENCODER_OUT][ACTOR])
        if self.model_config["action_type"] != "discrete":
            mean, log_std = model_out
            # TODO: Figure which return values to use
            # if continous we need a mean and log_std for a non-categorical distribution
            output[Columns.ACTION_DIST_INPUTS] = mean
            output[Columns.ACTION_LOGP] = log_std
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
        model_out = self.pi(encoder_outs[ENCODER_OUT][ACTOR])
        if self.model_config["action_type"] != "discrete":
            mean, log_std = model_out
            # TODO: Figure which return values to use
            output[Columns.ACTION_DIST_INPUTS] = mean
            output[Columns.ACTION_LOGP] = log_std
        else:
            output[Columns.ACTION_DIST_INPUTS] = model_out
        return output

    # endregion

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
            time dimension. Note that the last value dimension should already be
            squeezed out (not 1!).
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
        vf_out = self.vf(embeddings)  # type: ignore[arg-type]
        breakpoint()
        return vf_out.squeeze(-1)

    def _forward_inference(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Forward-pass used for action computation without exploration behavior.

        Override this method only, if you need specific behavior for non-exploratory
        action computation behavior. If you have only one generic behavior for all
        phases of training and evaluation, override `self._forward()` instead.

        By default, this calls the generic `self._forward()` method.
        """
        batch = jax.lax.stop_gradient(batch)
        return self._forward(batch, **kwargs)


if TYPE_CHECKING:
    SympolPPOModule(model_config={})
