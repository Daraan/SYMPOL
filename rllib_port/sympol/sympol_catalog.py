from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog

from ray_utilities.dummy_encoder import DummyActorCriticEncoderConfig
from rllib_port.mlp.mlp_model import ActorMLPContinuousModel, ActorMLPModel, CriticMLPModel
from rllib_port.sdt.sdt_model import ActorSDTModel, CriticSDTModel
from rllib_port.sympol.sympol_model import SympolRLModel

if TYPE_CHECKING:
    import gymnasium as gym


class SympolPPOCatalog(PPOCatalog):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict,
    ):
        super().__init__(observation_space, action_space, model_config_dict)
        self.actor_critic_encoder_config = DummyActorCriticEncoderConfig(
            base_encoder_config=self._encoder_config,
            shared=self._model_config_dict["vf_share_layers"],
        )
        self.actor_type: str = self._model_config_dict["actor"]

    def build_pi_head(self, framework: str) -> SympolRLModel | ActorMLPModel | ActorMLPContinuousModel | ActorSDTModel:
        if self.actor_type in ("mlp", "stateActionDT"):
            if self._model_config_dict["action_type"] == "discrete":
                actor = ActorMLPModel(
                    action_dim=self.action_space.n,  # type: ignore[attr-defined]
                    num_layers=self._model_config_dict["num_layers"],
                    neurons_per_layer=self._model_config_dict["neurons_per_layer"],
                )
            else:
                actor = ActorMLPContinuousModel(
                    action_dim=self.action_space.n,  # type: ignore[attr-defined]
                    num_layers=self._model_config_dict["num_layers"],
                    neurons_per_layer=self._model_config_dict["neurons_per_layer"],
                )
            # args.learning_rate_actor = args.learning_rate_critic  # same lr for MLP's
            # TODO: set this in setup!
            # args.accumulate_gradients_every = 1  # do not accumulate gradients for MLP's
            self._model_config_dict["accumulate_gradients_every"] = None  # likely no effect
            actor.apply = jax.jit(actor.apply)
        elif self.actor_type == "sympol":
            return SympolRLModel(
                obs_dim=self.observation_space.shape[0],  # type: ignore
                action_dim=self.action_space.n,  # type: ignore[attr-defined]
                depth=self._model_config_dict["depth"],
                n_estimators=self._model_config_dict["n_estimators"],
                action_type=self._model_config_dict["action_type"],
                subset_fraction=self._model_config_dict.get("subset_fraction", 0.8),
            )
        elif self.actor_type in ("sdt", "d-sdt"):
            actor = ActorSDTModel(
                action_dim=self.action_space.n,  # type: ignore[attr-defined]
                depth=self._model_config_dict["depth"],
                temperature=self._model_config_dict["temperature"],
                action_type=self._model_config_dict["action_type"],
            )  # , temp=1)

            # args.learning_rate_actor = args.learning_rate_critic  # same lr for SDT's
            # TODO:
            self._model_config_dict["accumulate_gradients_every"] = None  # do not accumulate gradients for SDT's
            actor.apply = jax.jit(actor.apply)
        else:
            raise ValueError(f"Actor '{self.actor_type}' not implemented")
        return actor

    def build_vf_head(self, framework: str) -> CriticMLPModel | CriticSDTModel:
        if self._model_config_dict["critic"] == "mlp":
            critic = CriticMLPModel(
                num_layers=self._model_config_dict["num_layers"],
                neurons_per_layer=self._model_config_dict["neurons_per_layer"],
            )
        elif self._model_config_dict["critic"] == "sdt":
            critic = CriticSDTModel(
                depth=self._model_config_dict["depth"],
                temperature=self._model_config_dict["temperature"],
            )
        else:
            raise NotImplementedError(f"Critic '{self._model_config_dict['critic']}' not implemented")
        critic.apply = jax.jit(critic.apply)
        return critic
