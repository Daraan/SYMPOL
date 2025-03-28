import logging
from typing import TYPE_CHECKING

from ray_utilities.jax.jax_model import FlaxRLModel
from sdt import Actor_SDT, Critic_SDT
from utils import _is_discreteT

logger = logging.getLogger(__name__)


class ActorSDTModel(Actor_SDT[_is_discreteT], FlaxRLModel):
    pass


class CriticSDTModel(Critic_SDT, FlaxRLModel):
    pass


if TYPE_CHECKING:
    ActorSDTModel(action_dim=0, depth=0, temperature=0.0, action_type="discrete")
    CriticSDTModel(depth=0, temperature=0.0)
