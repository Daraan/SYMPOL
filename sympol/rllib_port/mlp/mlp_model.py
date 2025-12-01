import logging
from typing import TYPE_CHECKING

import jax

from sympol.mlp import Actor_MLP, Actor_MLP_Continuous, Critic_MLP
from sympol.rllib_port.core.stated_flax_model import StatedActorFlaxRLModel, StatedCriticFlaxRLModel

if TYPE_CHECKING:
    from sympol.config_types.params_types import MLPParams

logger = logging.getLogger(__name__)


class ActorMLPModel(StatedActorFlaxRLModel[Actor_MLP, "MLPParams"]):
    def _setup_model(self, action_dim: int, **kwargs):  # noqa: ARG002
        model = Actor_MLP(
            action_dim=action_dim,
            num_layers=self.config["num_layers"],
            neurons_per_layer=self.config["neurons_per_layer"],
        )
        model.apply = jax.jit(model.apply)
        return model


class ActorMLPContinuousModel(StatedActorFlaxRLModel[Actor_MLP_Continuous, "MLPParams"]):
    def _setup_model(self, action_dim: int, **kwargs):  # noqa: ARG002
        model = Actor_MLP_Continuous(
            action_dim=action_dim,
            num_layers=self.config["num_layers"],
            neurons_per_layer=self.config["neurons_per_layer"],
        )
        model.apply = jax.jit(model.apply)
        return model


class CriticMLPModel(StatedCriticFlaxRLModel[Critic_MLP, "MLPParams"]):
    def _setup_model(self):
        model = Critic_MLP(
            num_layers=self.config["num_layers"],
            neurons_per_layer=self.config["neurons_per_layer"],
        )
        model.apply = jax.jit(model.apply)
        return model


if TYPE_CHECKING:
    # Check ABC
    ActorMLPModel(config={}, action_dim=1)  # pyright: ignore[reportArgumentType]
    ActorMLPContinuousModel(config={}, action_dim=1)  # pyright: ignore[reportArgumentType]
    CriticMLPModel(config={})  # pyright: ignore[reportArgumentType]
