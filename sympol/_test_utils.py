from __future__ import annotations

import sys
from dataclasses import asdict
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Optional, cast, overload
from unittest import mock

import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
import pytest
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig

from ray_utilities import DefaultTrainable
from ray_utilities.jax.jax_learner import JaxLearner
from ray_utilities.testing_utils import (
    _NOT_PROVIDED,
    DisableGUIBreakpoints,
    TestHelpers,
    get_explicit_required_keys,
    get_explicit_unrequired_keys,
    get_leafpath_value,
    get_optional_keys,
    get_required_keys,
    no_parallel_envs,
)
from ray_utilities.testing_utils import (
    SetupDefaults as _SetupDefaults,
)
from ray_utilities.testing_utils import (
    patch_args as _patch_args,
)
from sympol.config_types.args_types import SympolCLIArgs
from sympol.mlp import Actor_MLP, Critic_MLP
from sympol.rllib_port.core.sympol_module import SympolPPOModule
from sympol.rllib_port.extended_args import SympolArgumentParser
from sympol.rllib_port.sympol_setup import SympolSetup
from sympol.sdt import Actor_SDT, Critic_SDT
from sympol.sympol import SYMPOL_RL
from sympol.utils.utils import ActorTrainState, TrainState

if TYPE_CHECKING:
    from ray_utilities.jax.ppo.compute_ppo_loss import _PPOSettings
    from ray_utilities.training.default_class import TrainableBase
    from sympol.rllib_port.core.algorithms import SympolPPOConfig


if TYPE_CHECKING:
    import chex

    from ray_utilities.typing.metrics import AutoExtendedLogMetricsDict

__all__ = [
    "DisableGUIBreakpoints",
    "SympolSetupDefaults",
    "args_train_no_tuner",
    "clean_args",
    "get_explicit_required_keys",
    "get_explicit_unrequired_keys",
    "get_leafpath_value",
    "get_optional_keys",
    "get_required_keys",
    "sympol_patch_args",
]


_SympolTrainable = DefaultTrainable[SympolArgumentParser, PPOConfig, PPO]


def sympol_patch_args(*args, **kwargs):
    return _patch_args("--agent_type", "sympol", *args, **kwargs)


args_train_no_tuner = sympol_patch_args("-J", "1", "-it", "2", "-np")
clean_args = mock.patch.object(sys, "argv", ["file.py"])
"""Use when comparing to CLIArgs"""


patch_args = sympol_patch_args


class SympolTestHelpers(TestHelpers):
    def get_modules(self, trainable: TrainableBase):
        runner_module = cast("SympolPPOModule", trainable.algorithm.get_module())
        learner_module = cast("SympolPPOModule", trainable.algorithm.learner_group._learner.module["default_policy"])
        return runner_module, learner_module

    def compare_jax_learning_rate_set(self, trainable, target_lr: float, msg: str = ""):
        runner_module, learner_module = self.get_modules(trainable)

        # Check learner module
        learner_actor_state = learner_module.states["actor"]
        lrs_learner = JaxLearner._find_lrs(learner_actor_state.opt_state)
        self.assertTrue(len(lrs_learner) > 0, f"{msg} No hyperparameters found in learner opt_state")
        self.assertTrue(
            any(abs(lr - target_lr) < 1e-6 for lr in lrs_learner),
            f"{msg} Target LR {target_lr} not found in learner opt_state. Found: {optax.tree_utils.tree_get_all_with_path(learner_actor_state.opt_state, 'learning_rate')}",
        )

        # Check runner module
        runner_actor_state = runner_module.states["actor"]
        lrs_runner = JaxLearner._find_lrs(runner_actor_state.opt_state)
        self.assertTrue(len(lrs_runner) > 0, f"{msg} No hyperparameters found in runner opt_state")
        self.assertTrue(
            any(abs(lr - target_lr) < 1e-6 for lr in lrs_runner),
            f"{msg} Target LR {target_lr} not found in runner opt_state. Found: {optax.tree_utils.tree_get_all_with_path(runner_actor_state.opt_state, 'learning_rate')}",
        )

    def compare_config_attributes(
        self,
        config1: SympolPPOConfig | _PPOSettings,
        expected_valued: dict[str, Any],
        msg: str = "",
    ):
        for k, v in expected_valued.items():
            actual_value = getattr(config1, k)
            self.assertEqual(
                actual_value,
                v,
                f"{msg} PPO Config key '{k}' expected value {v}, got {actual_value}",
            )

    def check_grad_clip_set(self, trainable, target_clip: float, msg: str = ""):
        runner_module, learner_module = self.get_modules(trainable)

        for module_name, module in [("learner", learner_module), ("runner", runner_module)]:
            # module.pi is the actor (SympolRLModel)
            actor = module.pi
            config = actor.config

            # Handle potential key differences or defaults
            actual_clip = config.get("grad_clip")
            if actual_clip is None:
                raise KeyError("grad_clip not found, and max_grad_norm handling not implemented")

            self.assertIsNotNone(actual_clip, f"{msg} {module_name}: grad_clip not found in actor config")
            self.assertAlmostEqual(
                actual_clip,
                target_clip,
                places=6,
                msg=f"{msg} {module_name}: Expected grad_clip {target_clip}, got {actual_clip}",
            )

    def check_module_config_setting(self, trainable, key: str, expected: bool | Any = True):
        runner_module, learner_module = self.get_modules(trainable)

        for module_name, module in [("learner", learner_module), ("runner", runner_module)]:
            # module.pi is the actor (SympolRLModel)
            actor = module.pi
            config = actor.config
            # Cannot check actor_states directly as jit compiled

            self.assertIs(
                module.model_config.get(key),
                expected,
                f"{module_name}: Expected adamW to be {expected}, got {module.model_config.get('adamW')}",
            )

            actual_adamW = config.get(key)
            self.assertIsNotNone(actual_adamW, f"{module_name}: adamW not found in actor config")
            self.assertIs(
                actual_adamW,
                expected,
                f"{module_name}: Expected adamW to be {expected}, got {actual_adamW}",
            )

            if hasattr(module, "vf"):
                critic = module.vf
                critic_config = critic.config
                actual_adamW_critic = critic_config.get(key)
                self.assertIsNotNone(actual_adamW_critic, f"{module_name}: adamW not found in critic config")
                self.assertIs(
                    actual_adamW_critic,
                    expected,
                    f"{module_name}: Expected adamW to be {expected} in critic, got {actual_adamW_critic}",
                )
            else:
                assert module.inference_only

    @overload
    def get_trainable(
        self,
        *,
        num_env_runners: int = 0,
        env_seed: int | None | _NOT_PROVIDED = _NOT_PROVIDED,
        train: Any = True,
        fast_model: bool = True,
        eval_interval: Optional[int] = 1,
        class_only: Literal[True],
        ignore_argv: bool = True,
        setup_class=SympolSetup,
    ) -> type[_SympolTrainable]: ...

    @overload
    def get_trainable(
        self,
        *,
        num_env_runners: int = 0,
        env_seed: int | None | _NOT_PROVIDED = _NOT_PROVIDED,
        train: Literal[True] = True,
        fast_model: bool = True,
        eval_interval: Optional[int] = 1,
        class_only: Literal[False] = False,
        ignore_argv: bool = True,
        setup_class=SympolSetup,
    ) -> tuple[_SympolTrainable, AutoExtendedLogMetricsDict]: ...

    @overload
    def get_trainable(
        self,
        *,
        num_env_runners: int = 0,
        env_seed: int | None | _NOT_PROVIDED = _NOT_PROVIDED,
        train: Literal[False],
        fast_model: bool = True,
        eval_interval: Optional[int] = 1,
        class_only: Literal[False] = False,
        ignore_argv: bool = True,
        # setup_class = SympolSetup,
    ) -> tuple[_SympolTrainable, None]: ...

    def get_trainable(
        self,
        *,
        num_env_runners: int = 0,
        env_seed: int | None | _NOT_PROVIDED = 0,
        setup_class=SympolSetup,
        train: bool = True,
        depth: int = 2,
        class_only: bool = False,
        **setup_kwargs,
    ) -> (
        type[DefaultTrainable[SympolArgumentParser, PPOConfig, PPO]]
        | tuple[DefaultTrainable[SympolArgumentParser, PPOConfig, PPO], AutoExtendedLogMetricsDict | None]
    ):
        with _patch_args(
            "--agent_type",
            "sympol",
            "--depth", depth,
            #"--num_learners", 1,
        ):  # fmt: skip
            if class_only:
                return super().get_trainable(
                    num_env_runners=num_env_runners,
                    fast_model=False,
                    env_seed=env_seed,
                    ignore_argv=False,
                    setup_class=setup_class,
                    train=train,
                    class_only=True,
                    **setup_kwargs,
                )
            # weird type error if we do not check for train bool
            if train:
                trainable, result = super().get_trainable(
                    num_env_runners=num_env_runners,
                    fast_model=False,
                    env_seed=env_seed,
                    ignore_argv=False,
                    setup_class=setup_class,
                    class_only=False,
                    train=train,
                    **setup_kwargs,
                )
            else:
                trainable, result = super().get_trainable(
                    num_env_runners=num_env_runners,
                    fast_model=False,
                    env_seed=env_seed,
                    ignore_argv=False,
                    setup_class=setup_class,
                    class_only=False,
                    train=train,
                    **setup_kwargs,
                )
        module = trainable.algorithm.get_module()
        self.assertIsInstance(module, SympolPPOModule)
        return trainable, result  # pyright: ignore[reportReturnType]


class SympolSetupDefaults(_SetupDefaults, SympolTestHelpers):
    def setUp(self, setup_class=SympolSetup, *, empty_args=False):
        with _patch_args("--agent_type", "sympol"):
            super().setUp(setup_class=setup_class, empty_args=empty_args)
        print("Remember to enable/disable justMyCode('\"debugpy.debugJustMyCode\": false,') in the settings")
        env = gym.make("CartPole-v1")

        self._OBSERVATION_SPACE = env.observation_space
        self._ACTION_SPACE = env.action_space

        self._DEFAULT_CONFIG_DICT: Any = asdict(SympolCLIArgs())
        self._DEFAULT_CONFIG_DICT["action_dim"] = self._ACTION_SPACE.n  # type: ignore[attr-defined]
        self._DEFAULT_CONFIG_DICT["action_indices"] = list(range(self._DEFAULT_CONFIG_DICT["action_dim"]))
        self._DEFAULT_CONFIG_DICT = MappingProxyType(self._DEFAULT_CONFIG_DICT)
        self._DEFAULT_NAMESPACE = SympolCLIArgs()
        self._INPUT_LENGTH = env.observation_space.shape[0]  # pyright: ignore[reportOptionalSubscript]
        self._DEFAULT_INPUT = jnp.arange(self._INPUT_LENGTH * 2).reshape((2, self._INPUT_LENGTH))
        self._DEFAULT_BATCH: dict[str, chex.Array] = MappingProxyType({"obs": self._DEFAULT_INPUT})  # pyright: ignore[reportAttributeAccessIssue]
        self._ENV_SAMPLE = jnp.arange(self._INPUT_LENGTH)
        model_key = jax.random.PRNGKey(self._DEFAULT_CONFIG_DICT["seed"])
        self._RANDOM_KEY, self._ACTOR_KEY, self._CRITIC_KEY = jax.random.split(model_key, 3)
        self._ACTION_DIM: int = self._ACTION_SPACE.n  # type: ignore[attr-defined]
        self._OBS_DIM: int = self._OBSERVATION_SPACE.shape[0]  # pyright: ignore[reportOptionalSubscript]

    def _create_original_actor_state(self, model):
        return ActorTrainState.create(
            apply_fn=None,
            params=model.init(self._ACTOR_KEY, jnp.array([self._ENV_SAMPLE])),
            tx=optax.chain(
                optax.clip_by_global_norm(self._DEFAULT_NAMESPACE.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(self._DEFAULT_NAMESPACE.learning_rate_actor),
            ),
            grad_accum=jax.tree.map(jnp.zeros_like, model.init(self._ACTOR_KEY, jnp.array([self._ENV_SAMPLE]))),
            indices=None,
        )

    def _create_critic_state(self, critic):
        return TrainState.create(
            apply_fn=None,
            params=critic.init(self._CRITIC_KEY, jnp.array([self._ENV_SAMPLE])),
            tx=optax.chain(
                optax.clip_by_global_norm(self._DEFAULT_NAMESPACE.max_grad_norm),
                # adam or adamW:
                (optax.adamw if self._DEFAULT_NAMESPACE.adamW else optax.adam)(
                    learning_rate=self._DEFAULT_NAMESPACE.learning_rate_critic
                ),
            ),
        )

    def _create_actor_sdt(self):
        model = Actor_SDT(
            self._ACTION_DIM,
            depth=self._DEFAULT_NAMESPACE.depth,
            temperature=self._DEFAULT_NAMESPACE.temperature,
            action_type=self._DEFAULT_NAMESPACE.action_type,
        )
        state = self._create_original_actor_state(model)
        return model, state

    def _create_critic_mlp(self):
        model = Critic_MLP(
            num_layers=self._DEFAULT_NAMESPACE.num_layers,
            neurons_per_layer=self._DEFAULT_NAMESPACE.neurons_per_layer,
        )
        state = self._create_critic_state(model)
        return model, state

    def _create_actor_mlp(self):
        model = Actor_MLP(
            action_dim=self._ACTION_DIM,
            num_layers=self._DEFAULT_NAMESPACE.num_layers,
            neurons_per_layer=self._DEFAULT_NAMESPACE.neurons_per_layer,
        )
        state = self._create_original_actor_state(model)
        return model, state

    def _create_critic_sdt(self):
        model = Critic_SDT(
            depth=self._DEFAULT_NAMESPACE.depth,
            temperature=self._DEFAULT_NAMESPACE.temperature,
        )
        state = self._create_critic_state(model)
        return model, state

    def _create_actor_sympol(self):
        actor = SYMPOL_RL(
            obs_dim=self._INPUT_LENGTH,
            action_dim=self._ACTION_DIM,
            depth=self._DEFAULT_NAMESPACE.depth,
            n_estimators=self._DEFAULT_NAMESPACE.n_estimators,
            action_type=self._DEFAULT_NAMESPACE.action_type,
        )

        def map_nested_fn(fn):
            """Recursively apply `fn` to key-value pairs of a nested dict."""

            def map_fn(nested_dict):
                return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v)) for k, v in nested_dict.items()}

            return map_fn

        args = self._DEFAULT_NAMESPACE
        actor_key = self._ACTOR_KEY

        # original code
        if args.SWA:
            from optax_swag import swag

            if args.adamW:
                actor_state = ActorTrainState.create(
                    apply_fn=None,
                    params=actor.init(actor_key, jnp.array([self._ENV_SAMPLE])),
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
                    grad_accum=jax.tree.map(jnp.zeros_like, actor.init(actor_key, jnp.array([self._ENV_SAMPLE]))),
                    indices=actor.init_indices(actor_key) if args.actor == "sympol" else None,
                )
            else:
                actor_state = ActorTrainState.create(
                    apply_fn=None,
                    params=actor.init(actor_key, jnp.array([self._ENV_SAMPLE])),
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
                    grad_accum=jax.tree.map(jnp.zeros_like, actor.init(actor_key, jnp.array([self._ENV_SAMPLE]))),
                    indices=actor.init_indices(actor_key) if args.actor == "sympol" else None,
                )
        else:  # noqa
            if args.adamW:
                actor_state: ActorTrainState = ActorTrainState.create(
                    apply_fn=None,
                    params=actor.init(actor_key, jnp.array([self._ENV_SAMPLE])),
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
                    ),
                    grad_accum=jax.tree.map(jnp.zeros_like, actor.init(actor_key, jnp.array([self._ENV_SAMPLE]))),
                    indices=actor.init_indices(actor_key) if args.actor == "sympol" else None,
                )
            else:
                actor_state = ActorTrainState.create(
                    apply_fn=None,
                    params=actor.init(actor_key, jnp.array([self._ENV_SAMPLE])),
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
                    ),
                    grad_accum=jax.tree.map(jnp.zeros_like, actor.init(actor_key, jnp.array([self._ENV_SAMPLE]))),
                    indices=actor.init_indices(actor_key) if args.actor == "sympol" else None,
                )
        return actor, actor_state
