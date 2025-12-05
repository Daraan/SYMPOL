from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING

import gymnasium as gym
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray_utilities.dummy_encoder import DummyActorCriticEncoderConfig
from ray_utilities.jax.catalog.jax_catalog import JaxCatalog

from sympol.rllib_port.mlp.mlp_model import ActorMLPContinuousModel, ActorMLPModel, CriticMLPModel
from sympol.rllib_port.sdt.sdt_model import ActorSDTModel, CriticSDTModel
from sympol.rllib_port.sympol.sympol_model import SympolRLModel

if TYPE_CHECKING:
    from sympol.config_types.params_types import SympolCatalogOptions

logger = logging.getLogger(__name__)


class SympolJaxPPOCatalog(JaxCatalog, PPOCatalog):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: SympolCatalogOptions,
    ):
        super().__init__(observation_space, action_space, model_config_dict)  # pyright: ignore[reportArgumentType]
        self.actor_critic_encoder_config = DummyActorCriticEncoderConfig(
            base_encoder_config=self._encoder_config,
            shared=self._model_config_dict.get("vf_share_layers", True),
        )
        self._actor_type = model_config_dict["actor"]
        self._critic_type = model_config_dict["critic"]

        self._action_dist_class_fn = functools.partial(
            self._get_dist_cls_from_action_space, action_space=self.action_space
        )

        self._model_config_dict: SympolCatalogOptions

    def build_pi_head(self, framework: str) -> SympolRLModel | ActorMLPModel | ActorMLPContinuousModel | ActorSDTModel:  # noqa: ARG002
        # action_dim and action_indices must be set in config before catalog construction
        action_dim = self._model_config_dict.get("action_dim")
        action_indices = self._model_config_dict.get("action_indices")
        if action_dim is None or action_indices is None:
            raise ValueError(
                "action_dim and action_indices must be set in model_config_dict before catalog construction."
            )
        if self._actor_type in ("mlp", "stateActionDT"):
            assert self.action_space.n == action_dim
            if self._model_config_dict["action_type"] == "discrete":
                actor = ActorMLPModel(
                    config=self._model_config_dict,
                    action_dim=self.action_space.n,  # pyright: ignore[reportAttributeAccessIssue]
                )
            else:
                actor = ActorMLPContinuousModel(
                    config=self._model_config_dict,
                    action_dim=self.action_space.n,  # pyright: ignore[reportAttributeAccessIssue]
                )
            # args.learning_rate_actor = args.learning_rate_critic  # same lr for MLP's
        elif self._actor_type == "sympol":
            assert self.observation_space.shape is not None
            return SympolRLModel(
                obs_dim=self.observation_space.shape[0],
                action_dim=action_dim,
                config=self._model_config_dict,
            )
        elif self._actor_type in ("sdt", "d-sdt"):
            actor = ActorSDTModel(
                config=self._model_config_dict,
                action_dim=self.action_space.n,  # pyright: ignore[reportAttributeAccessIssue]
            )  # , temp=1)

            # args.learning_rate_actor = args.learning_rate_critic  # same lr for SDT's
        else:
            raise ValueError(f"Actor '{self._actor_type}' not implemented")
        return actor

    def build_vf_head(self, framework: str) -> CriticMLPModel | CriticSDTModel:
        if self._critic_type == "mlp":
            critic = CriticMLPModel(self._model_config_dict)
        elif self._critic_type == "sdt":
            critic = CriticSDTModel(self._model_config_dict)
        else:
            raise NotImplementedError(f"Critic '{self._critic_type}' not implemented")
        return critic
