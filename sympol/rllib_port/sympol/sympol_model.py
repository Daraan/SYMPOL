from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional, overload

import jax
import jax.numpy as jnp
import optax
from ray.rllib.utils.typing import TensorType
from typing_extensions import Self

from ray_utilities.jax.jax_model import JaxRLModel, PureJaxModelProtocol
from sympol.sympol import SYMPOL_RL
from sympol.utils.utils import ActorTrainState

if TYPE_CHECKING:
    import chex
    from chex import Array
    from flax.core.frozen_dict import FrozenDict
    from flax.typing import FrozenVariableDict
    from ray.rllib.utils.typing import TensorType

    from sympol.config_types.params_types import SYMPOLModelArgsDict, SympolParams

logger = logging.getLogger(__name__)


# NOTE: Not a pytree node
class SympolRLModel(JaxRLModel):
    if TYPE_CHECKING:

        def __config_type(self):  # noqa
            self.config: SympolParams

        @overload
        def __call__(
            self, obs: "Array", *, parameters: "FrozenDict", indices: Any
        ) -> "Array | tuple[Array, Array]": ...

        @overload
        def __call__(self, obs: "Array", *, parameters: "FrozenDict", **kwargs) -> "Array | tuple[Array, Array]": ...

    def __unsqueeze_log_std(self, log_std: "Array", target_dim: int) -> "Array":
        """Unsqueeze log_std to match target_dim."""
        while log_std.ndim < target_dim:
            log_std = log_std[jnp.newaxis]
        return log_std

    def _forward(
        self, input_dict, *, parameters, indices: FrozenVariableDict | Mapping, **kwargs
    ) -> tuple[chex.Array, chex.Array] | jax.Array:
        out = super()._forward(input_dict, parameters=parameters, indices=indices, **kwargs)
        if self.config["action_type"] != "discrete":
            mean, log_std = out
            return mean, self.__unsqueeze_log_std(log_std, target_dim=mean.ndim)
        return out

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

    def __repr__(self):
        return f"""{self.__class__.__name__}(
    config={self.config},
    model={self.model!r}
    )
    """

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

        gradient_transformations: list[optax.GradientTransformation] = [
            # grad_clip is rllib max_grad_norm is legacy
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
        ]
        grad_clip = config["grad_clip"] if "grad_clip" in config else config["max_grad_norm"]
        if grad_clip is not None:
            gradient_transformations.insert(0, optax.clip_by_global_norm(grad_clip))
        if config["SWA"]:
            from optax_swag import swag

            gradient_transformations.append(swag(10, 2))

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
