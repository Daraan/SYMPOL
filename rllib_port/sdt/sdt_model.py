import logging
from typing import TYPE_CHECKING

from sdt import Actor_SDT, Critic_SDT

from rllib_port.model_interface import FlaxRLInterface

logger = logging.getLogger(__name__)


class ActorSDTModel(Actor_SDT, FlaxRLInterface):
    pass


class CriticSDTModel(Critic_SDT, FlaxRLInterface):
    pass


if TYPE_CHECKING:
    ActorSDTModel(action_dim=0, depth=0, temperature=0.0, action_type="discrete")
    CriticSDTModel(depth=0, temperature=0.0)
