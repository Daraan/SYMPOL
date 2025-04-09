from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp
import optax
from ray.rllib.utils.typing import TensorType
from typing_extensions import Self

from ray_utilities.jax.jax_model import JaxRLModel, PureJaxModelProtocol
from sympol import SYMPOL_RL
from utils.utils import ActorTrainState

if TYPE_CHECKING:
    import chex
    from ray.rllib.utils.typing import TensorType

    from config_types.params_types import SYMPOLModelArgsDict, SympolParams

logger = logging.getLogger(__name__)


# NOTE: Not a pytree node
class SympolRLModel(JaxRLModel):
    if TYPE_CHECKING:

        def __config_type(self):  # noqa
            self.config: SympolParams

        def __call__(self, *args, **kwargs) -> TensorType:
            """Call the model."""
            return super().__call__(*args, **kwargs)

    def __init__(self, *, obs_dim: int, action_dim: int, config: SYMPOLModelArgsDict):
        JaxRLModel.__init__(self, config=config)  # type: ignore[arg-type] not a ModelConfig
        # pytree_node_class
        self.model = SYMPOL_RL(
            obs_dim=obs_dim,
            action_dim=action_dim,
            depth=config["depth"],
            n_estimators=config["n_estimators"],
            action_type=config["action_type"],
            subset_fraction=config.get("subset_fraction", 0.8),
        )
        assert self.config == config

    def init_state(
        self: Self | PureJaxModelProtocol,
        rng: chex.PRNGKey,
        sample: TensorType | chex.Array,
        *,
        config: Optional[SympolParams] = None,
    ) -> ActorTrainState:
        """
        Arg:
            sample: envs.single_observation_space.sample()
            rng: The actor_key
        """
        if isinstance(self, PureJaxModelProtocol):
            actor = self
            if config is None:
                raise ValueError("If using init_state as a classmethod, config should be passed")
        else:
            actor = self.model
            if config is None:
                config = self.config

        def map_nested_fn(fn):
            """Recursively apply `fn` to key-value pairs of a nested dict."""

            def map_fn(nested_dict):
                return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v)) for k, v in nested_dict.items()}

            return map_fn

        if config["adamW"]:
            optimizer = optax.adamw
        else:
            optimizer = optax.adam

        gradient_transformations = (
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.multi_transform(
                {
                    "estimator_weights": optax.inject_hyperparams(optax.adam)(config["learning_rate_actor_weights"]),
                    "split_values": optax.inject_hyperparams(optax.adam)(config["learning_rate_actor_split_values"]),
                    "split_idx_array": optax.inject_hyperparams(optimizer)(
                        config["learning_rate_actor_split_idx_array"]
                    ),
                    "leaf_array": optax.inject_hyperparams(optimizer)(config["learning_rate_actor_leaf_array"]),
                    "log_std": optax.inject_hyperparams(optimizer)(config["learning_rate_actor_log_std"]),
                },
                map_nested_fn(lambda k, _: k),
            ),
        )
        if config["SWA"]:
            from optax_swag import swag

            gradient_transformations = (*gradient_transformations, swag(10, 2))

        actor_state = ActorTrainState.create(
            apply_fn=None,
            params=actor.init(rng, jnp.array([sample])),
            tx=optax.chain(*gradient_transformations),
            grad_accum=jax.tree.map(jnp.zeros_like, actor.init(rng, jnp.array([sample]))),
            indices=actor.init_indices(rng),
        )
        return actor_state


if TYPE_CHECKING:
    SympolRLModel(obs_dim=0, action_dim=0, config=SYMPOLModelArgsDict(**{}))  # noqa: PIE804
