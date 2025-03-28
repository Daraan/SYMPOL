import logging
from typing import TYPE_CHECKING

from mlp import Actor_MLP, Actor_MLP_Continuous, Critic_MLP
from ray_utilities.jax.jax_model import FlaxRLModel

logger = logging.getLogger(__name__)


class ActorMLPModel(Actor_MLP, FlaxRLModel):
    pass


class ActorMLPContinuousModel(Actor_MLP_Continuous, FlaxRLModel):
    pass


class CriticMLPModel(Critic_MLP, FlaxRLModel):
    pass


if TYPE_CHECKING:
    ActorMLPModel(0, 0, 0)
    ActorMLPContinuousModel(0, 0, 0)
    CriticMLPModel(0, 0)
