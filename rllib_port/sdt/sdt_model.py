from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import jax

from rllib_port.core.stated_flax_model import StatedActorFlaxRLModel, StatedCriticFlaxRLModel
from sdt import Actor_SDT, Critic_SDT

if TYPE_CHECKING:
    from config_types.params_types import SDTParams
    from utils import _is_discreteT

logger = logging.getLogger(__name__)


class ActorSDTModel(StatedActorFlaxRLModel[Actor_SDT["_is_discreteT"], "SDTParams"]):
    def _setup_model(self, action_dim: int, **kwargs) -> Actor_SDT[_is_discreteT]:
        model: Actor_SDT[_is_discreteT] = Actor_SDT(
            action_dim=action_dim,
            depth=self.config["depth"],
            temperature=self.config["temperature"],
            action_type=self.config["action_type"],
            **kwargs,
        )
        # possibly add indices as static arg
        model.apply = jax.jit(model.apply)
        return model


class CriticSDTModel(StatedCriticFlaxRLModel[Critic_SDT, "SDTParams"]):
    def _setup_model(self, **kwargs) -> Critic_SDT:
        model = Critic_SDT(
            depth=self.config["depth"],
            temperature=self.config["temperature"],
            **kwargs,
        )
        model.apply = jax.jit(model.apply)
        return model


if TYPE_CHECKING:
    # Check ABC
    ActorSDTModel(action_dim=0, config=SDTParams(**{}))  # noqa: PIE804
    CriticSDTModel(SDTParams(**{}))  # noqa: PIE804
