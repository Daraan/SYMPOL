import logging
from typing import TYPE_CHECKING

from mlp import Actor_MLP, Actor_MLP_Continuous, Critic_MLP

from rllib_port.model_interface import FlaxRLInterface

logger = logging.getLogger(__name__)


class ActorMLPModel(Actor_MLP, FlaxRLInterface):
    pass


class ActorMLPContinuousModel(Actor_MLP_Continuous, FlaxRLInterface):
    pass


class CriticMLPModel(Critic_MLP, FlaxRLInterface):
    pass


if TYPE_CHECKING:
    ActorMLPModel(0, 0, 0)
    ActorMLPContinuousModel(0, 0, 0)
    CriticMLPModel(0, 0)
