from __future__ import annotations

import distrax
import jax.numpy as jnp
import numpy as np

from typing import cast, TYPE_CHECKING
from utils import ActorTrainState, build_env
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from utils.utils import ObservationActionBuffer, Storage

if TYPE_CHECKING:
    from mlp import Actor_MLP_Continuous
    import jax
    import gymnasium as gym
    from mlp import Actor_MLP
    from args import CLIArgs


def fit_stateActionDT(
    actor_state: ActorTrainState,
    env_id: str,
    n_episodes: int,
    name_appendix,
    seed: int = 1_000,
    *,
    envs: gym.vector.VectorEnv,
    action_dim: int,  # todo might be inferable from envs, depends on custom envs
    n_steps: int,
    args: CLIArgs,
    actor: Actor_MLP | Actor_MLP_Continuous,
    storage: Storage,
    action_indices: list[int],
) -> DecisionTreeClassifier | DecisionTreeRegressor | list[DecisionTreeRegressor]:
    assert envs.single_observation_space.shape is not None
    assert envs.single_action_space.shape is not None
    action_obs_store = ObservationActionBuffer(
        # obs=jnp.zeros((n_steps, args.n_envs) + envs.single_observation_space.shape),
        obs=jnp.zeros((n_steps, n_episodes, *envs.single_observation_space.shape)),
        # actions=jnp.zeros((n_steps, args.n_envs) + envs.single_action_space.shape,
        actions=jnp.zeros((n_steps, n_episodes, *envs.single_action_space.shape), dtype=jnp.int32),
    )

    total_eval_steps = 0
    for episode_index in range(n_episodes):
        # temp_env = build_env(env_id, n_env=1)
        temp_env = build_env(env_id, n_env=1, view_size=args.view_size)

        done, trunc = False, False
        obs, info = temp_env.reset(seed=seed + episode_index)  # random.randint(0, 1000))
        step_counter = 0
        while not done and not trunc:
            actor_params = actor_state.params

            obs = np.array([obs]).reshape((-1, envs.single_observation_space.shape[0]))
            if args.action_type == "discrete":
                action_logits = cast("jax.Array", actor.apply(actor_params, obs, indices=actor_state.indices))
                action = jnp.argmax(action_logits, axis=1)
                action = jnp.squeeze(
                    action, axis=0
                )  # jnp.squeeze(action, axis=0) if action.shape[0] == 1 else action #action[0]
            else:
                result = actor.apply(actor_params, obs, indices=actor_state.indices)
                action_distribution = distrax.MultivariateNormalDiag(result[0], jnp.exp(result[1]))  # pyright: ignore[reportArgumentType]
                action = action_distribution.mean()
                action = jnp.squeeze(action, axis=0)

            action_obs_store = action_obs_store.replace(
                obs=storage.obs.at[total_eval_steps].set(obs),
                actions=storage.actions.at[total_eval_steps].set(action),
            )

            action = np.array(action)
            if any(
                substring in args.env_id
                for substring in [
                    "MultiRoom",
                    "Unlock",
                    "GoToDoor",
                    "UnlockPickup",
                    "DoorKey",
                    "RedBlueDoors",
                ]
            ):
                action = action_indices[action]
            next_obs, rewards, done, trunc, info = temp_env.step(action)

            obs = next_obs
            step_counter += 1
            total_eval_steps += 1

        temp_env.close()

    # Initialize decision tree
    decision_tree: DecisionTreeClassifier | DecisionTreeRegressor | list[DecisionTreeRegressor]
    if args.action_type == "discrete":
        decision_tree = DecisionTreeClassifier(max_depth=args.depth)
    else:  # noqa
        if action_dim == 1:
            decision_tree = DecisionTreeRegressor(max_depth=args.depth)
        else:
            decision_tree = [DecisionTreeRegressor(max_depth=args.depth) for _ in range(action_dim)]

    # Train the decision tree
    X = np.array(action_obs_store.obs).reshape(-1, temp_env.observation_space.shape[-1])  # pyright: ignore[reportPossiblyUnboundVariable, reportOptionalSubscript]

    if args.action_type == "discrete" or action_dim == 1:
        y = np.array(action_obs_store.actions).reshape(-1)
        decision_tree.fit(X, y)  # type: ignore[attr-defined]
    else:
        for i in range(action_dim):
            y = np.array(action_obs_store.actions).reshape(-1, action_dim)
            decision_tree[i].fit(X, y[:, i])  # pyright: ignore[reportIndexIssue]

    return decision_tree
