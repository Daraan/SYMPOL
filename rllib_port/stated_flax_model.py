from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from typing_extensions import TypeVar

from ray_utilities.jax.jax_model import FlaxRLModel, ModelType
from utils.utils import ActorTrainState

if TYPE_CHECKING:
    import chex
    from ray.rllib.utils.typing import TensorType

    from config_types.params_types import GeneralParamsWithLR

logger = logging.getLogger(__name__)

_ConfigType = TypeVar("_ConfigType", bound="GeneralParamsWithLR", default="GeneralParamsWithLR")


class StatedActorFlaxRLModel(FlaxRLModel[ModelType, _ConfigType]):
    @abstractmethod
    def _setup_model(self, action_dim: int, **kwargs) -> ModelType:
        """Set up the underlying flax model."""
        ...

    def __init__(self, config: _ConfigType, *, action_dim: int, **kwargs):
        super().__init__(config=config, action_dim=action_dim, **kwargs)

    def init_state(
        self,
        rng: chex.PRNGKey,
        sample: TensorType | chex.Array,
        *,
        config: Optional[_ConfigType] = None,
    ) -> ActorTrainState:
        actor_key = rng
        if config is None:
            config = self.config
        if config.get("adamW", False):
            logger.warning("adamW=True is set for a non SYMPOL actor. Check if correct")
        optimizer_cls = optax.adamw if config.get("adamW", False) else optax.adam
        actor_state = ActorTrainState.create(
            apply_fn=None,
            params=self.model.init(actor_key, jnp.array([sample])),
            tx=optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.inject_hyperparams(optimizer_cls)(config["learning_rate_actor"]),
            ),
            grad_accum=jax.tree.map(jnp.zeros_like, self.model.init(actor_key, jnp.array([sample]))),
            indices=None,
        )
        return actor_state


class StatedCriticFlaxRLModel(FlaxRLModel[ModelType, _ConfigType]):
    def init_state(
        self,
        rng: chex.PRNGKey,
        sample: TensorType | chex.Array,
        *,
        config: Optional[GeneralParamsWithLR] = None,
    ) -> TrainState:
        if config is None:
            config = self.config
        optimizer = optax.adamw if config.get("adamW", False) else optax.adam
        critic_state = TrainState.create(
            apply_fn=None,
            params=self.model.init(rng, jnp.array([sample])),
            tx=optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optimizer(learning_rate=config["learning_rate_critic"]),
            ),
        )
        return critic_state
