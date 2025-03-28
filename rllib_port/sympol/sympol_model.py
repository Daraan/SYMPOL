from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax

from ray_utilities.jax.jax_model import JaxRLModel
from sympol import SYMPOL_RL
from utils.utils import ActorTrainState

if TYPE_CHECKING:
    import chex
    from ray.rllib.utils.typing import TensorType

    from config_types.params_types import SYMPOLModelArgsDict, SympolParams

logger = logging.getLogger(__name__)


class SympolRLModel(SYMPOL_RL, JaxRLModel):
    if TYPE_CHECKING:

        def __config_type(self):  # noqa
            self.config: SympolParams

    def __init__(self, *, obs_dim: int, action_dim: int, config: SYMPOLModelArgsDict):
        JaxRLModel.__init__(self, config)  # type: ignore[arg-type] not a ModelConfig
        config["action_type"]
        SYMPOL_RL.__init__(
            self,
            obs_dim=obs_dim,
            action_dim=action_dim,
            depth=config["depth"],
            n_estimators=config["n_estimators"],
            action_type=config["action_type"],
            subset_fraction=config.get("subset_fraction", 0.8),
        )

    def init_state(self, rng: chex.PRNGKey, sample: TensorType | chex.Array) -> ActorTrainState:
        """
        Arg:
            sample: envs.single_observation_space.sample()
            rng: The actor_key
        """
        actor = self

        def map_nested_fn(fn):
            """Recursively apply `fn` to key-value pairs of a nested dict."""

            def map_fn(nested_dict):
                return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v)) for k, v in nested_dict.items()}

            return map_fn

        if self.config["SWA"]:
            from optax_swag import swag

            if self.config["adamW"]:
                actor_state = ActorTrainState.create(
                    apply_fn=None,
                    params=actor.init(rng, jnp.array([sample])),
                    tx=optax.chain(
                        optax.clip_by_global_norm(self.config["max_grad_norm"]),
                        optax.multi_transform(
                            {
                                "estimator_weights": optax.inject_hyperparams(optax.adam)(
                                    self.config["learning_rate_actor_weights"]
                                ),
                                "split_values": optax.inject_hyperparams(optax.adam)(
                                    self.config["learning_rate_actor_split_values"]
                                ),
                                "split_idx_array": optax.inject_hyperparams(optax.adamw)(
                                    self.config["learning_rate_actor_split_idx_array"]
                                ),
                                "leaf_array": optax.inject_hyperparams(optax.adamw)(
                                    self.config["learning_rate_actor_leaf_array"]
                                ),
                                "log_std": optax.inject_hyperparams(optax.adamw)(
                                    self.config["learning_rate_actor_log_std"]
                                ),
                            },
                            map_nested_fn(lambda k, _: k),
                        ),
                        swag(10, 2),
                    ),
                    grad_accum=jax.tree.map(jnp.zeros_like, actor.init(rng, jnp.array([sample]))),
                    indices=actor.init_indices(rng),
                )
            else:
                actor_state = ActorTrainState.create(
                    apply_fn=None,
                    params=actor.init(rng, jnp.array([sample])),
                    tx=optax.chain(
                        optax.clip_by_global_norm(self.config["max_grad_norm"]),
                        optax.multi_transform(
                            {
                                "estimator_weights": optax.inject_hyperparams(optax.adam)(
                                    self.config["learning_rate_actor_weights"]
                                ),
                                "split_values": optax.inject_hyperparams(optax.adam)(
                                    self.config["learning_rate_actor_split_values"]
                                ),
                                "split_idx_array": optax.inject_hyperparams(optax.adam)(
                                    self.config["learning_rate_actor_split_idx_array"]
                                ),
                                "leaf_array": optax.inject_hyperparams(optax.adam)(
                                    self.config["learning_rate_actor_leaf_array"]
                                ),
                                "log_std": optax.inject_hyperparams(optax.adam)(
                                    self.config["learning_rate_actor_log_std"]
                                ),
                            },
                            map_nested_fn(lambda k, _: k),
                        ),
                        swag(10, 2),
                    ),
                    grad_accum=jax.tree.map(jnp.zeros_like, actor.init(rng, jnp.array([sample]))),
                    indices=actor.init_indices(rng),
                )
        elif self.config["adamW"]:
            actor_state: ActorTrainState = ActorTrainState.create(
                apply_fn=None,
                params=actor.init(rng, jnp.array([sample])),
                tx=optax.chain(
                    optax.clip_by_global_norm(self.config["max_grad_norm"]),
                    optax.multi_transform(
                        {
                            "estimator_weights": optax.inject_hyperparams(optax.adam)(
                                self.config["learning_rate_actor_weights"]
                            ),
                            "split_values": optax.inject_hyperparams(optax.adam)(
                                self.config["learning_rate_actor_split_values"]
                            ),
                            "split_idx_array": optax.inject_hyperparams(optax.adamw)(
                                self.config["learning_rate_actor_split_idx_array"]
                            ),
                            "leaf_array": optax.inject_hyperparams(optax.adamw)(
                                self.config["learning_rate_actor_leaf_array"]
                            ),
                            "log_std": optax.inject_hyperparams(optax.adamw)(
                                self.config["learning_rate_actor_log_std"]
                            ),
                        },
                        map_nested_fn(lambda k, _: k),
                    ),
                ),
                grad_accum=jax.tree.map(jnp.zeros_like, actor.init(rng, jnp.array([sample]))),
                indices=actor.init_indices(rng),
            )
        else:
            actor_state = ActorTrainState.create(
                apply_fn=None,
                params=actor.init(rng, jnp.array([sample])),
                tx=optax.chain(
                    optax.clip_by_global_norm(self.config["max_grad_norm"]),
                    optax.multi_transform(
                        {
                            "estimator_weights": optax.inject_hyperparams(optax.adam)(
                                self.config["learning_rate_actor_weights"]
                            ),
                            "split_values": optax.inject_hyperparams(optax.adam)(
                                self.config["learning_rate_actor_split_values"]
                            ),
                            "split_idx_array": optax.inject_hyperparams(optax.adam)(
                                self.config["learning_rate_actor_split_idx_array"]
                            ),
                            "leaf_array": optax.inject_hyperparams(optax.adam)(
                                self.config["learning_rate_actor_leaf_array"]
                            ),
                            "log_std": optax.inject_hyperparams(optax.adam)(self.config["learning_rate_actor_log_std"]),
                        },
                        map_nested_fn(lambda k, _: k),
                    ),
                ),
                grad_accum=jax.tree.map(jnp.zeros_like, actor.init(rng, jnp.array([sample]))),
                indices=actor.init_indices(rng),
            )
        return actor_state


if TYPE_CHECKING:
    SympolRLModel(obs_dim=0, action_dim=0, config=SYMPOLModelArgsDict(**{}))  # noqa: PIE804
