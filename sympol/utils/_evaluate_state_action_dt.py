from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional, cast

import distrax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

import wandb
from sympol.utils.envs import _select_action, build_env
from sympol.utils.envs.drawing import draw_env
from sympol.utils.trees import convert_to_discrete_tree
from sympol.utils.trees.drawing import plot_decision_tree
from sympol.utils.utils import OBSERVATION_LABELS, ActorTrainState

if TYPE_CHECKING:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    from sympol.config_types.args_types import SympolCLIArgs
    from sympol.sdt import Actor_SDT

logger = logging.getLogger(__name__)


def evaluate_state_action_dt(
    episode_index: int,
    env_id: str,
    name_appendix: str,
    seed: int = 100,
    decision_tree: Optional[DecisionTreeClassifier | DecisionTreeRegressor | list[DecisionTreeRegressor]] = None,
    *,
    video_folder: str = "videos/wandb",
    args: SympolCLIArgs,
    action_dim: int,
    run_name: str,
    render_now: bool,
    action_indices: list[int],
) -> tuple[list[float], int | None]:
    # temp_env = build_env(env_id, n_env=1)
    score_interpretable = []
    node_count: int | None = 0
    temp_env = build_env(env_id, n_env=1, view_size=args.view_size)
    video_path = os.path.join(video_folder, run_name + "-" + "-" + env_id + str(episode_index) + ".mp4")
    image_path = os.path.join(video_folder, run_name + "-" + "-" + env_id)  #  + str(episode_index))

    done, trunc = False, False
    obs, info = temp_env.reset(seed=seed + episode_index)  # random.randint(0, 1000))
    running_reward = 0
    frames = []
    dones = False
    step_counter = 0
    while not done and not trunc:
        if args.render_env and render_now:
            if (image := draw_env(temp_env, step_counter, running_reward)) is not None:
                frames.append(image)

        flat_obs = obs.reshape(1, -1)
        assert decision_tree is not None
        if args.action_type == "discrete":
            decision_tree = cast("DecisionTreeClassifier | DecisionTreeRegressor", decision_tree)  # noqa: E501
            action = decision_tree.predict(flat_obs)[0]
        else:  # noqa
            if action_dim == 1:
                action = decision_tree.predict(flat_obs)  # type: ignore[attr-defined]
            else:
                action_list = []
                for i in range(action_dim):
                    action_by_tree = decision_tree[i].predict(flat_obs)[0]  # pyright: ignore[reportIndexIssue]
                    action_list.append(action_by_tree)
                action = np.array(action_list)

        action = _select_action(env_id, action, action_indices)
        next_obs, rewards, done, trunc, info = temp_env.step(action)

        running_reward += rewards  # type: ignore[operator]

        obs = next_obs
        step_counter += 1

    score_interpretable.append(running_reward)

    if args.render_env and render_now:
        if False:
            if (image := draw_env(temp_env, step_counter, running_reward)) is not None:
                frames.append(image)

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

        if episode_index == 0:
            if args.action_type == "discrete" or action_dim == 1:
                decision_tree = cast("DecisionTreeClassifier | DecisionTreeRegressor", decision_tree)  # noqa: E501
                # Plot the decision tree
                plt.figure(figsize=(20, 10))
                plot_tree(decision_tree, filled=True)
                plt.title("Decision Tree")

                image_path = os.path.join(video_folder, run_name + "-" + "-" + args.env_id)
                plot_filename = image_path + "state_action_DT.png"
                plt.savefig(plot_filename)
                plt.close()
                if args.track:
                    # Log the image to wandb
                    wandb.log({"state_action_DT": wandb.Image(plot_filename)})
                # This probably always 0+ node_count
                if node_count is None:
                    node_count = decision_tree.tree_.node_count
                else:
                    node_count += decision_tree.tree_.node_count
            else:
                # TODO: is this repeated too often?
                node_count = 0
                decision_tree = cast("list[DecisionTreeRegressor]", decision_tree)
                for i in range(action_dim):
                    # Plot the decision tree
                    plt.figure(figsize=(20, 10))
                    plot_tree(decision_tree[i], filled=True)
                    plt.title("Decision Tree " + str(i))

                    image_path = os.path.join(video_folder, run_name + "-" + "-" + args.env_id)
                    plot_filename = image_path + "state_action_DT_reg" + str(i) + ".png"
                    plt.savefig(plot_filename)
                    plt.close()
                    node_count += decision_tree[i].tree_.node_count
                    if args.track:
                        # Log the image to wandb
                        wandb.log({"state_action_DT_" + str(i): wandb.Image(plot_filename)})

    temp_env.close()
    return score_interpretable, node_count


def evaluate_state_d_sdt(
    episode_index: int,
    env_id: str,
    name_appendix: str,
    seed: int = 100,
    *,
    video_folder: str = "videos/wandb",
    args: SympolCLIArgs,
    obs_dim: int,
    run_name: str,
    render_now: bool,
    action_indices: list[int],
    actor: Actor_SDT,
    actor_state: ActorTrainState,
) -> tuple[list[float], int | None]:
    # temp_env = build_env(env_id, n_env=1)
    temp_env = build_env(env_id, n_env=1, view_size=args.view_size)
    video_path = os.path.join(video_folder, run_name + "-" + "-" + env_id + str(episode_index) + ".mp4")
    image_path = os.path.join(video_folder, run_name + "-" + "-" + env_id)  #  + str(episode_index))

    done, trunc = False, False
    obs, info = temp_env.reset(seed=seed + episode_index)  # random.randint(0, 1000))
    running_reward = 0.0
    frames = []
    score_interpretable: list[float] = []
    step_counter = 0
    # Prevent unbound variable
    actor_params_discrete = {}
    actor_params = {}
    while not done and not trunc:
        if args.render_env and render_now:
            if (image := draw_env(temp_env, step_counter, running_reward)) is not None:
                frames.append(image)

        actor_params = actor_state.params
        actor_params_discrete = convert_to_discrete_tree(actor_params, args.action_type, temperature=args.temperature)
        if args.action_type == "discrete":
            action_logits = actor.apply(
                actor_params_discrete,
                np.array([obs]).reshape((-1, obs_dim)),
                max_path=True,
                indices=actor_state.indices,
            )

            action = jnp.argmax(action_logits, axis=1)  # pyright: ignore[reportArgumentType]
            action = jnp.squeeze(
                action, axis=0
            )  # jnp.squeeze(action, axis=0) if action.shape[0] == 1 else action #action[0]
        else:
            result = actor.apply(
                actor_params_discrete,
                np.array([obs]).reshape((-1, obs_dim)),
                max_path=True,
                indices=actor_state.indices,
            )

            action_distribution = distrax.MultivariateNormalDiag(result[0], jnp.exp(result[1]))  # pyright: ignore[reportArgumentType]
            action = action_distribution.mean()
            action = jnp.squeeze(action, axis=0)

        action = np.array(action)

        action = _select_action(env_id, action, action_indices)
        next_obs, rewards, done, trunc, info = temp_env.step(action)

        running_reward += rewards  # type: ignore[operator]

        obs = next_obs
        step_counter += 1

    score_interpretable.append(running_reward)
    node_count = None
    if args.render_env and render_now:
        if False:
            if (image := draw_env(temp_env, step_counter, running_reward)) is not None:
                frames.append(image)

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
                # wandb_log["gameplay_" + name_appendix + '_trial' + str(episode_index)] = wandb.Video(numpy_clip, fps=fps, format="mp4")  # noqa: E501
        if episode_index == 0:
            split_values = actor_params_discrete["params"]["SDT_0"]["inner_nodes"]["layers_0"][
                "kernel"
            ].T * jnp.expand_dims(  # pyright: ignore[reportAttributeAccessIssue]
                actor_params_discrete["params"]["SDT_0"]["inner_nodes"]["layers_0"]["bias"], 1
            )
            split_indices = actor_params_discrete["params"]["SDT_0"]["inner_nodes"]["layers_0"]["kernel"].T  # pyright: ignore[reportAttributeAccessIssue]
            leaf_values = actor_params_discrete["params"]["SDT_0"]["leaf_nodes"]["kernel"]

            image_path, node_count = plot_decision_tree(
                split_values=split_values,
                split_indices=split_indices,
                leaf_values=leaf_values,
                features_by_estimator=list(range(obs_dim)),
                image_path=image_path,
                observation_labels=None
                if args.env_id not in OBSERVATION_LABELS.keys()
                else OBSERVATION_LABELS[args.env_id],
                filename_appendix="D-SDT",
                env=temp_env,
                # env_params=None,
                prune=True,
                continuous=args.action_type != "discrete",
            )
            image_path_plot = image_path + ".png"
            if args.track:
                wandb.log(
                    {"D-SDT_" + name_appendix + "_trial" + str(episode_index): wandb.Image(image_path_plot)},
                    commit=False,
                )

            image_path_complete = image_path + "_COMPLETE"
            image_path_complete, _ = plot_decision_tree(
                split_values=split_values,
                split_indices=split_indices,
                leaf_values=leaf_values,
                features_by_estimator=list(range(obs_dim)),
                image_path=image_path_complete,
                observation_labels=None
                if args.env_id not in OBSERVATION_LABELS.keys()
                else OBSERVATION_LABELS[args.env_id],
                filename_appendix="D-SDT",
                env=temp_env,
                # env_params=None,
                prune=False,
                continuous=args.action_type != "discrete",
            )

            image_path_plot = image_path_complete + ".png"
            if args.track:
                wandb.log(
                    {"D-SDT_COMPLETE" + name_appendix + "_trial" + str(episode_index): wandb.Image(image_path_plot)},
                    commit=False,
                )

    temp_env.close()
    return (score_interpretable, node_count)
