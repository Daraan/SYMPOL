from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import jax.numpy as jnp
import optax
from flax import struct

from ray_utilities.jax.jax_model import JaxRLModel
from sympol import SYMPOL_RL
from utils.utils import ActorTrainState


if TYPE_CHECKING:
    from config_types.args_types import CLIArgs
    import chex
    from ray.rllib.utils.typing import TensorType

logger = logging.getLogger(__name__)


@struct.dataclass(kw_only=True, frozen=False)
class SympolRLModel(SYMPOL_RL, JaxRLModel):

    def init_state(self, rng: chex.PRNGKey, sample: TensorType | chex.Array) -> ActorTrainState:
        """
        Arg:
            sample: envs.single_observation_space.sample()
            rng: The actor_key
        """
        actor = self

        if TYPE_CHECKING:
            args = CLIArgs()
        def map_nested_fn(fn):
            """Recursively apply `fn` to key-value pairs of a nested dict."""

            def map_fn(nested_dict):
                return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v)) for k, v in nested_dict.items()}

            return map_fn

        if args.SWA:
            from optax_swag import swag

            if args.adamW:
                actor_state = ActorTrainState.create(
                    apply_fn=None,
                    params=actor.init(rng, jnp.array([sample])),
                    tx=optax.chain(
                        optax.clip_by_global_norm(args.max_grad_norm),
                        optax.multi_transform(
                            {
                                "estimator_weights": optax.inject_hyperparams(optax.adam)(
                                    args.learning_rate_actor_weights
                                ),
                                "split_values": optax.inject_hyperparams(optax.adam)(
                                    args.learning_rate_actor_split_values
                                ),
                                "split_idx_array": optax.inject_hyperparams(optax.adamw)(
                                    args.learning_rate_actor_split_idx_array
                                ),
                                "leaf_array": optax.inject_hyperparams(optax.adamw)(
                                    args.learning_rate_actor_leaf_array
                                ),
                                "log_std": optax.inject_hyperparams(optax.adamw)(args.learning_rate_actor_log_std),
                            },
                            map_nested_fn(lambda k, _: k),
                        ),
                        swag(10, 2),
                    ),
                    grad_accum=jax.tree.map(
                        jnp.zeros_like, actor.init(rng, jnp.array([sample]))
                    ),
                    indices=actor.init_indices(rng) if args.actor == "sympol" else None,
                )
            else:
                actor_state = ActorTrainState.create(
                    apply_fn=None,
                    params=actor.init(rng, jnp.array([sample])),
                    tx=optax.chain(
                        optax.clip_by_global_norm(args.max_grad_norm),
                        optax.multi_transform(
                            {
                                "estimator_weights": optax.inject_hyperparams(optax.adam)(
                                    args.learning_rate_actor_weights
                                ),
                                "split_values": optax.inject_hyperparams(optax.adam)(
                                    args.learning_rate_actor_split_values
                                ),
                                "split_idx_array": optax.inject_hyperparams(optax.adam)(
                                    args.learning_rate_actor_split_idx_array
                                ),
                                "leaf_array": optax.inject_hyperparams(optax.adam)(args.learning_rate_actor_leaf_array),
                                "log_std": optax.inject_hyperparams(optax.adam)(args.learning_rate_actor_log_std),
                            },
                            map_nested_fn(lambda k, _: k),
                        ),
                        swag(10, 2),
                    ),
                    grad_accum=jax.tree.map(
                        jnp.zeros_like, actor.init(rng, jnp.array([sample]))
                    ),
                    indices=actor.init_indices(rng) if args.actor == "sympol" else None,
                )
        elif args.adamW:
            actor_state: ActorTrainState = ActorTrainState.create(
                apply_fn=None,
                params=actor.init(rng, jnp.array([sample])),
                tx=optax.chain(
                    optax.clip_by_global_norm(args.max_grad_norm),
                    optax.multi_transform(
                        {
                            "estimator_weights": optax.inject_hyperparams(optax.adam)(args.learning_rate_actor_weights),
                            "split_values": optax.inject_hyperparams(optax.adam)(args.learning_rate_actor_split_values),
                            "split_idx_array": optax.inject_hyperparams(optax.adamw)(
                                args.learning_rate_actor_split_idx_array
                            ),
                            "leaf_array": optax.inject_hyperparams(optax.adamw)(
                                args.learning_rate_actor_leaf_array
                            ),
                            "log_std": optax.inject_hyperparams(optax.adamw)(
                                args.learning_rate_actor_log_std
                            ),
                        },
                        map_nested_fn(lambda k, _: k),
                    ),
                ),
                grad_accum=jax.tree.map(
                    jnp.zeros_like, actor.init(rng, jnp.array([sample]))
                ),
                indices=actor.init_indices(rng) if args.actor == "sympol" else None,
            )
        else:
            actor_state = ActorTrainState.create(
                apply_fn=None,
                params=actor.init(rng, jnp.array([sample])),
                tx=optax.chain(
                    optax.clip_by_global_norm(args.max_grad_norm),
                    optax.multi_transform(
                        {
                            "estimator_weights": optax.inject_hyperparams(optax.adam)(args.learning_rate_actor_weights),
                            "split_values": optax.inject_hyperparams(optax.adam)(args.learning_rate_actor_split_values),
                            "split_idx_array": optax.inject_hyperparams(optax.adam)(
                                args.learning_rate_actor_split_idx_array
                            ),
                            "leaf_array": optax.inject_hyperparams(optax.adam)(args.learning_rate_actor_leaf_array),
                            "log_std": optax.inject_hyperparams(optax.adam)(args.learning_rate_actor_log_std),
                        },
                        map_nested_fn(lambda k, _: k),
                    ),
                ),
                grad_accum=jax.tree.map(
                    jnp.zeros_like, actor.init(rng, jnp.array([sample]))
                ),
                indices=actor.init_indices(rng) if args.actor == "sympol" else None,
            )
        return actor_state


if TYPE_CHECKING:
    SympolRLModel(obs_dim=0, action_dim=0, action_type="discrete", depth=1, n_estimators=1)
