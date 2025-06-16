from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional, cast

import distrax
import jax.numpy as jnp
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import wandb
from utils import OBSERVATION_LABELS, ActorTrainState, build_env
from utils._evaluate_state_action_dt import evaluate_state_action_dt, evaluate_state_d_sdt
from utils.envs import _select_action
from utils.envs.drawing import draw_env
from utils.trees.drawing import plot_decision_tree, plot_decision_tree_soft
from utils.type_guard import is_d_sdt, is_sdt_actor, is_sympol_actor

if TYPE_CHECKING:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    from config_types.args_types import SympolCLIArgs
    from mlp import Actor_MLP, Actor_MLP_Continuous, Critic_MLP
    from ray_utilities.jax.jax_model import PureJaxModelProtocol
    from sdt import Actor_SDT, Critic_SDT
    from sympol import SYMPOL_RL

    _Actor = PureJaxModelProtocol | Actor_MLP | Actor_MLP_Continuous | Actor_SDT | SYMPOL_RL
    _Critic = Critic_MLP | Critic_SDT


def evaluate_agent(
    actor_state: ActorTrainState,
    env_id: str,
    n_episodes: int,
    name_appendix: str,
    seed: int = 100,
    decision_tree: Optional[DecisionTreeClassifier | DecisionTreeRegressor | list[DecisionTreeRegressor]] = None,
    *,
    args: SympolCLIArgs,
    action_dim: int,
    run_name: str,
    render_now: bool,
    action_indices: list[int],
    actor: _Actor,
    obs_dim: int,
) -> tuple[list[float], list[float], int]:
    video_folder = "videos/wandb"
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    # temp_env = Monitor(temp_env, video_folder) #, force=True

    score: list[float] = []
    score_interpretable = []
    node_count = 0
    for episode_index in range(n_episodes):
        if args.actor == "stateActionDT":
            next_score_interpretable, next_node_count = evaluate_state_action_dt(
                episode_index,
                env_id,
                name_appendix,
                seed,
                decision_tree,
                video_folder=video_folder,
                args=args,
                action_dim=action_dim,
                run_name=run_name,
                render_now=render_now,
                action_indices=action_indices,
            )
        elif is_d_sdt(actor, args):
            next_score_interpretable, next_node_count = evaluate_state_d_sdt(
                episode_index,
                env_id,
                name_appendix,
                seed,
                video_folder=video_folder,
                args=args,
                obs_dim=obs_dim,
                run_name=run_name,
                render_now=render_now,
                action_indices=action_indices,
                actor=actor,
                actor_state=actor_state,
            )
        else:
            next_node_count = 0
            next_score_interpretable = []
        if episode_index == 0:
            # NOTE: args.render_env and render_now are necessary to get a node_count != None
            node_count = next_node_count
        score_interpretable.extend(next_score_interpretable)

        # temp_env = build_env(env_id, n_env=1)
        temp_env = build_env(env_id, n_env=1, view_size=args.view_size)
        video_path = os.path.join(video_folder, run_name + "-" + "-" + env_id + str(episode_index) + ".mp4")
        image_path = os.path.join(video_folder, run_name + "-" + "-" + env_id)  #  + str(episode_index))

        done, trunc = False, False
        obs, info = temp_env.reset(seed=seed + episode_index)  # random.randint(0, 1000))
        running_reward = 0
        frames = []
        step_counter = 0
        while not done and not trunc:
            if args.render_env and render_now:
                frames.append(draw_env(temp_env, step_counter, running_reward))

            actor_params = actor_state.params
            obs = np.array([obs]).reshape((-1, obs_dim))
            if args.action_type == "discrete":
                action_logits = actor.apply(actor_params, obs, indices=actor_state.indices)

                action = jnp.argmax(action_logits, axis=1)  # type: ignore[arg-type]
                action = jnp.squeeze(
                    action, axis=0
                )  # jnp.squeeze(action, axis=0) if action.shape[0] == 1 else action #action[0]
            else:
                result = actor.apply(actor_params, obs, indices=actor_state.indices)
                action_distribution = distrax.MultivariateNormalDiag(result[0], jnp.exp(result[1]))  # pyright: ignore[reportIndexIssue, reportArgumentType]
                action = action_distribution.mean()
                action = jnp.squeeze(action, axis=0)

            action = np.array(action)

            action = _select_action(env_id, action, action_indices)
            next_obs, rewards, done, trunc, info = temp_env.step(action)

            running_reward += rewards  # type: ignore[operator]

            obs = next_obs
            step_counter += 1
        if TYPE_CHECKING:
            actor_params = cast("dict[str, Any]", actor_params)  # pyright: ignore[reportPossiblyUnboundVariable]

        score.append(running_reward)
        if args.render_env and render_now:
            if False:
                frames.append(draw_env(temp_env, obs, step_counter, running_reward))

                numpy_clip = np.transpose(np.array(frames), (0, 3, 1, 2))
                fps = 5 if "MiniGrid" in env_id else 25
                if args.track:
                    wandb.log(
                        {
                            "gameplay_" + name_appendix + "_trial" + str(episode_index): wandb.Video(
                                numpy_clip, fps=fps, format="mp4"
                            )
                        },
                        commit=False,
                    )
                    # wandb_log["gameplay_" + name_appendix + '_trial' + str(episode_index)] = wandb.Video(numpy_clip, fps=fps, format="mp4")

            if args.n_estimators <= 5 and is_sympol_actor(actor, args) and episode_index == 0:
                for estimator_number in range(args.n_estimators):
                    filename_appendix = "_" + str(estimator_number)
                    image_path, node_count = plot_decision_tree(
                        split_values=actor_params["split_values"][estimator_number],
                        split_indices=actor_params["split_idx_array"][estimator_number],
                        leaf_values=actor_params["leaf_array"][estimator_number],
                        features_by_estimator=actor_state.indices["features_by_estimator"][estimator_number],
                        image_path=image_path,
                        observation_labels=None
                        if args.env_id not in OBSERVATION_LABELS.keys()
                        else OBSERVATION_LABELS[args.env_id],
                        filename_appendix=filename_appendix,
                        env=temp_env,
                        # env_params=None,
                        prune=True,
                        continuous=args.action_type != "discrete",
                    )
                image_path_plot = image_path + ".png"
                if args.track:
                    wandb.log(
                        {
                            "DT_"
                            + name_appendix
                            + "_trial"
                            + str(episode_index)
                            + "_estNumber"
                            + str(estimator_number): wandb.Image(image_path_plot)  # pyright: ignore[reportPossiblyUnboundVariable]
                        },
                        commit=False,
                    )
                for estimator_number in range(args.n_estimators):
                    filename_appendix = "_" + str(estimator_number)
                    image_path_complete = image_path + "_COMPLETE"
                    image_path_complete, _ = plot_decision_tree(
                        split_values=actor_params["split_values"][estimator_number],
                        split_indices=actor_params["split_idx_array"][estimator_number],
                        leaf_values=actor_params["leaf_array"][estimator_number],
                        features_by_estimator=actor_state.indices["features_by_estimator"][estimator_number],
                        image_path=image_path_complete,
                        observation_labels=None
                        if args.env_id not in OBSERVATION_LABELS.keys()
                        else OBSERVATION_LABELS[args.env_id],
                        filename_appendix=filename_appendix,
                        env=temp_env,
                        # env_params=None,
                        prune=False,
                        continuous=args.action_type != "discrete",
                    )

                image_path_plot = image_path_complete + ".png"  # type: ignore
                if args.track:
                    wandb.log(
                        {
                            "DT_COMPLETE"
                            + name_appendix
                            + "_trial"
                            + str(episode_index)
                            + "_estNumber"
                            + str(estimator_number): wandb.Image(image_path_plot)  # pyright: ignore[reportPossiblyUnboundVariable]
                        },
                        commit=False,
                    )
            elif is_sdt_actor(actor, args) and episode_index == 0:
                split_values = actor_params["params"]["SDT_0"]["inner_nodes"]["layers_0"]["bias"]
                split_indices = actor_params["params"]["SDT_0"]["inner_nodes"]["layers_0"]["kernel"].T  # type: ignore[attr-defined]
                leaf_values = actor_params["params"]["SDT_0"]["leaf_nodes"]["kernel"]

                image_path, node_count_sdt = plot_decision_tree_soft(
                    split_values=split_values,
                    split_indices=split_indices,
                    leaf_values=leaf_values,
                    image_path=image_path,
                    observation_labels=None
                    if args.env_id not in OBSERVATION_LABELS.keys()
                    else OBSERVATION_LABELS[args.env_id],
                    filename_appendix="SDT",
                    temperature=args.temperature,
                )
                if args.actor == "sdt":  # else d-sdt
                    node_count = node_count_sdt
                image_path_plot = image_path + ".png"
                if args.track:
                    wandb.log(
                        {"SDT_" + name_appendix + "_trial" + str(episode_index): wandb.Image(image_path_plot)},
                        commit=False,
                    )

        temp_env.close()

    # avg_score = np.mean(score)
    # avg_score_interpretable = np.mean(score_interpretable)
    # return avg_score.item(), avg_score_interpretable.item(), node_count
    if node_count is None:
        node_count = 0
    return score, score_interpretable, node_count
