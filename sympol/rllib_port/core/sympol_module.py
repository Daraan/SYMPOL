from __future__ import annotations

# pyright: reportIncompatibleMethodOverride=warning
# pyright: reportIncompatibleVariableOverride=warning
import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union, cast

import jax
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT
from ray.rllib.core.rl_module.apis import InferenceOnlyAPI

from ray_utilities.jax.distributions.get_distributions_mixin import GetJaxDistributionsMixin
from ray_utilities.jax.jax_module import JaxModuleState
from ray_utilities.jax.ppo.jax_ppo_module import JaxActorCriticStateDict, JaxPPOModule
from sympol.rllib_port.core.sympol_catalog import SympolJaxPPOCatalog
from sympol.rllib_port.mlp.mlp_model import CriticMLPModel
from sympol.utils.get_action_and_value import get_action_and_value

if TYPE_CHECKING:
    import chex
    import gymnasium as gym
    from flax.core import FrozenDict
    from numpy.typing import NDArray
    from ray.rllib.utils.typing import StateDict

    from ray_utilities.dummy_encoder import DummyActorCriticEncoder
    from sympol.config_types.params_types import CLIArgsDict, SympolCatalogOptions
    from sympol.rllib_port.mlp.mlp_model import ActorMLPContinuousModel, ActorMLPModel, CriticMLPModel
    from sympol.rllib_port.sdt.sdt_model import ActorSDTModel, CriticSDTModel
    from sympol.rllib_port.sympol.sympol_model import SympolRLModel
    from sympol.utils.utils import ActorTrainState, Storage, TrainState


# for a Intermediate old API to new API Module

logger = logging.getLogger(__name__)


class SympolPPOStateDict(JaxActorCriticStateDict):
    actor: ActorTrainState  # pyright: ignore[reportIncompatibleVariableOverride]


class SympolModuleState(JaxModuleState):
    model_config: SympolCatalogOptions | StateDict  # pyright: ignore[reportIncompatibleVariableOverride]


class SympolPPOModule(GetJaxDistributionsMixin, JaxPPOModule):
    # torch code: which should be equivalent
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
        self.model_config: SympolCatalogOptions
        self.states: SympolPPOStateDict  # pyright: ignore[reportIncompatibleVariableOverride]
        super().__init__(
            config=config,  # deprecated
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

        self.pi: SympolRLModel | ActorMLPModel | ActorMLPContinuousModel | ActorSDTModel[bool]
        self.vf: CriticSDTModel | CriticMLPModel
        self.encoder: DummyActorCriticEncoder

    def to(self, device: Optional[chex.Device] = None):
        # FIXME: Implement proper device handling
        return
        if device is None:
            # put all states on device
            device = jax.local_devices()[0]
        self.states[ACTOR] = jax.device_put(self.states[ACTOR], device)
        self.states[CRITIC] = jax.device_put(self.states[CRITIC], device)

    # region: forward methods

    # Analog to DefaultPPOTorchRLModule
    def _forward(
        self, batch: dict[str, Any], *, parameters: Optional[Mapping] = None, indices: Optional[dict] = None, **kwargs
    ) -> dict[str, Any]:
        """
        Default forward pass (used for inference and exploration).

        Note:
            To compute gradients pass parameters via keyword `parameters`.
        """
        output = {}
        # Encoder forward pass.
        encoder_outs = self.encoder(batch)
        # Stateful encoder?
        if Columns.STATE_OUT in encoder_outs:
            output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]  # pyright: ignore[reportGeneralTypeIssues]  # key is present
        # Pi head.
        model_out = self.pi(
            encoder_outs[ENCODER_OUT][ACTOR],
            parameters=parameters if parameters is not None else self.states[ACTOR].params,
            indices=indices if indices is not None else self.states[ACTOR].indices,
        )
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
    def _forward_train(self, batch: dict[str, Any], *, parameters, indices=None, **kwargs) -> dict[str, Any]:
        """
        Train forward pass (keep embeddings for possible shared value func. call).

        Note:
            To compute gradients pass parameters via keyword `parameters`.
        """
        output = {}
        encoder_outs = self.encoder(batch)
        output[Columns.EMBEDDINGS] = encoder_outs[ENCODER_OUT][CRITIC]  # pyright: ignore[reportTypedDictNotRequiredAccess]  # training
        if Columns.STATE_OUT in encoder_outs:
            output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]  # pyright: ignore[reportGeneralTypeIssues]  # key is present
        model_out = self.pi(
            encoder_outs[ENCODER_OUT][ACTOR],
            parameters=parameters,
            # As indices are kind of constant we can use this fallback
            indices=self.states[ACTOR].indices if indices is None else indices,
            **kwargs,
        )
        if self.model_config["action_type"] != "discrete":
            # mean, log_std = model_out
            # TODO: Figure which return values to use
            output[Columns.ACTION_DIST_INPUTS] = model_out
        else:
            output[Columns.ACTION_DIST_INPUTS] = model_out
        return output

    # endregion

    def get_state(
        self,
        *args,  # noqa: ARG002
        inference_only: bool = False,
        **kwargs,  # noqa: ARG002
    ) -> SympolModuleState:
        state_dict = self.states
        # critic state not needed; possibly only bother when using GPU
        # however, if we copy the dict -> key updates are not performed -> repeated usage of keys!
        if inference_only and not self.inference_only:
            state_dict = state_dict.copy()
            attr = [*self.get_non_inference_attributes(), "critic"]
            for key in list(state_dict.keys()):
                if any(key.startswith(a) and (len(key) == len(a) or key[len(a)] == ".") for a in attr):
                    del state_dict[key]
        return {"jax_state": state_dict, "model_config": self.model_config}

    def set_state(self, state: JaxModuleState | StateDict | SympolModuleState) -> None:
        if "model_config" in state:
            state = state.copy()
            new_config = state.get("model_config")
            build_new = None
            if new_config and self.model_config != new_config:
                self.model_config = new_config
                self.catalog = type(self.catalog)(self.observation_space, self.action_space, new_config)  # pyright: ignore[reportArgumentType]
                self.setup()
                build_new = False
            if self.model_config:
                # with catalog update these all should now be fine.
                if self.pi.config != self.pi.config | self.model_config:
                    # model needs update
                    logger.info("Updating pi model config in set_state")
                    # self.pi.config = self.pi.config.update(self.model_config)
                    # should rebuild it entirely
                    build_new = True
                if hasattr(self, "vf") and self.vf.config != self.vf.config | self.model_config:
                    logger.info("Updating vf model config in set_state")
                    build_new = True
                if build_new:
                    # setup catalog just to be sure
                    self.catalog = type(self.catalog)(self.observation_space, self.action_space, self.model_config)  # pyright: ignore[reportArgumentType]
                    self.setup()
                    assert self.pi.config == self.pi.config | self.model_config
                    assert not hasattr(self, "vf") or self.vf.config == self.vf.config | self.model_config
            # TODO need to updates models
        super().set_state(state)

    # region non-rllib interface

    def update_state(self, *, actor: Optional[ActorTrainState], critic: Optional[TrainState]):
        """Update the actor and critic states."""
        if actor:
            self.states[ACTOR] = actor
        if critic:
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

    def parameters(self) -> tuple[FrozenDict[str, jax.Array], FrozenDict[str, jax.Array]]:
        return self.states[ACTOR].params, self.states[CRITIC].params


if TYPE_CHECKING:
    SympolPPOModule(model_config={})
