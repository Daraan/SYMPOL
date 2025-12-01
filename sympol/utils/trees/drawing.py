from __future__ import annotations
import logging
from types import SimpleNamespace

import gymnasium as gym

import numpy as np
from typing import Any, MutableMapping, Optional, cast, TYPE_CHECKING
from sympol.utils.trees import (
    convert_to_child_representation,
    convert_to_child_representation_soft,
    count_nodes,
    prune_and_merge_tree,
)

if TYPE_CHECKING:
    from ray_utilities.typing import EnvType

_logger = logging.getLogger(__name__)

__all__ = [
    "plot_decision_tree",
    "plot_decision_tree_soft",
    "plot_tree_from_representation",
    "plot_tree_from_representation_soft",
]


def plot_tree_from_representation_soft(
    tree: MutableMapping[str, Any], image_path: str, filename_appendix: str = "", observation_labels=None
) -> str:
    def add_nodes_edges(tree, dot=None):
        if dot is None:
            import graphviz

            dot = graphviz.Digraph()

        def traverse(node: MutableMapping[str, Any], parent=None):
            if node["type"] == "leaf":
                label = f"Action: {node['action']}"
                node_id = str(id(node))
                dot.node(node_id, label, shape="box")
            else:
                if np.round(node["split_value"]) == 1 and node["split_value"] < 1:
                    node["split_value"] = 0.99
                elif np.round(node["split_value"]) == -1 and node["split_value"] > -1:
                    node["split_value"] = -0.99
                label = f"{np.round(node['split_index'], 2)} - {node['split_value']:.2f}?"
                node_id = str(id(node))
                dot.node(node_id, label)
                traverse(node["left_child"], node_id)
                dot.edge(node_id, str(id(node["left_child"])), label="True")
                traverse(node["right_child"], node_id)
                dot.edge(node_id, str(id(node["right_child"])), label="False")

        traverse(tree)
        return dot

    dot = add_nodes_edges(tree)
    image_path = image_path + filename_appendix
    dot.render(image_path, format="png", cleanup=True)
    return image_path


def plot_decision_tree_soft(
    split_values, split_indices, leaf_values, image_path, observation_labels=None, filename_appendix="", temperature=1.0
) -> tuple[str, int]:
    tree_representation = convert_to_child_representation_soft(split_values, split_indices, leaf_values, temperature)

    node_count = count_nodes(tree_representation)
    plot_path = plot_tree_from_representation_soft(
        tree_representation, image_path, filename_appendix="", observation_labels=observation_labels
    )

    return plot_path, node_count["internal"] + node_count["leaf"]


def plot_tree_from_representation(tree, image_path, filename_appendix="", observation_labels=None, continuous=False):
    def add_nodes_edges(tree, dot=None):
        if dot is None:
            import graphviz

            dot = graphviz.Digraph()

        def traverse(node, parent=None):
            if node["type"] == "leaf":
                label = f"Action: {node['action']}" if continuous else f"Action: {np.argmax(node['action'])}"
                node_id = str(id(node))
                dot.node(node_id, label, shape="box")
            else:
                if np.round(node["split_value"]) == 1 and node["split_value"] < 1:
                    node["split_value"] = 0.99
                elif np.round(node["split_value"]) == -1 and node["split_value"] > -1:
                    node["split_value"] = -0.99
                if observation_labels is not None:
                    label = f"{observation_labels[node['split_index']]} <= {node['split_value']:.2f}?"
                else:
                    label = f"X{node['split_index']} <= {node['split_value']:.2f}?"
                node_id = str(id(node))
                dot.node(node_id, label)
                traverse(node["left_child"], node_id)
                dot.edge(node_id, str(id(node["left_child"])), label="True")
                traverse(node["right_child"], node_id)
                dot.edge(node_id, str(id(node["right_child"])), label="False")

        traverse(tree)
        return dot

    dot = add_nodes_edges(tree)
    image_path = image_path + filename_appendix
    dot.render(image_path, format="png", cleanup=True)
    return image_path


def plot_decision_tree(
    split_values,
    split_indices,
    leaf_values,
    features_by_estimator,
    image_path,
    observation_labels=None,
    filename_appendix="",
    env: Optional[EnvType] = None,
    env_params=None,
    *,
    prune=True,
    continuous=False,
) -> tuple[str, int]:
    tree_representation = convert_to_child_representation(
        split_values, split_indices, leaf_values, features_by_estimator
    )
    if prune:
        assert env
        ranges_dict = None
        if env_params is not None:
            env = cast("environment_gymnax.Environment", env)
            env_name = env.name
            observation_space = env.observation_space(env_params)
            try:
                from gymnax.environments import environment as environment_gymnax, spaces as spaces_gymnax
            except ImportError:

                class Dummy:
                    pass

                spaces_gymnax = SimpleNamespace(Box=Dummy, Discrete=Dummy)
            if isinstance(observation_space, spaces_gymnax.Box):
                ranges_dict = {}
                for i, range_tuple in enumerate(np.vstack([observation_space.low, observation_space.high]).T):
                    ranges_dict[i] = list(np.asarray(range_tuple))
                print(ranges_dict)
            elif isinstance(observation_space, spaces_gymnax.Discrete):
                ranges_dict = {}
                for i in range(observation_space.n):
                    ranges_dict[i] = [0, 1]
                print(ranges_dict)
            else:
                print("Observation Space type is not handled in this snippet.")
        else:
            env = cast("gym.Env", env)
            observation_space = env.observation_space
            if isinstance(env, gym.vector.SyncVectorEnv):
                env_name = env.envs[0].unwrapped.spec.id  # type: ignore[attr-defined]
            else:
                env_name = env.unwrapped.spec.id  # type: ignore[attr-defined]
            if "MiniGrid" in env_name:
                observation_space = cast("gym.spaces.Box", observation_space)  # best guess
                ranges_dict = {}
                for i, _range_tuple in enumerate(np.vstack([observation_space.low, observation_space.high]).T):
                    ranges_dict[i] = [-1, 1]
                print(ranges_dict)
            else:
                if isinstance(observation_space, gym.spaces.Box):
                    ranges_dict = {}
                    for i, range_tuple in enumerate(np.vstack([observation_space.low, observation_space.high]).T):
                        ranges_dict[i] = list(range_tuple)
                    print(ranges_dict)
                elif isinstance(observation_space, gym.spaces.Discrete):
                    ranges_dict = {}
                    for i in range(observation_space.n):
                        ranges_dict[i] = [0, 1]
                    print(ranges_dict)
                else:
                    print("Observation Space type is not handled in this snippet.")
        if ranges_dict is None:
            _logger.error("Unsupported setting for env and its observation space. This might cause an error.")
            ranges_dict = {}  # try to continue instead of raising an error
        tree_representation = prune_and_merge_tree(tree_representation, ranges_dict, continuous=continuous)
    node_count = count_nodes(tree_representation)
    plot_path = plot_tree_from_representation(
        tree_representation,
        image_path,
        filename_appendix="",
        observation_labels=observation_labels,
        continuous=continuous,
    )

    return plot_path, node_count["internal"] + node_count["leaf"]
