from __future__ import annotations

import copy
from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze

from utils.jax_math import entmax15JAX

__all__ = [
    "convert_to_child_representation",
    "convert_to_child_representation_soft",
    "convert_to_discrete_tree",
    "count_nodes",
    "prune_and_merge_tree",
]


def convert_to_child_representation_soft(split_values, split_indices, leaf_values, temperature):
    num_internal_nodes = split_values.shape[0]
    num_leaf_nodes = leaf_values.shape[0]

    def build_tree(node_id):
        if node_id >= num_internal_nodes:
            leaf_index = node_id - num_internal_nodes
            leaf_dist = leaf_values[leaf_index]
            return {"type": "leaf", "action": np.argmax(leaf_dist), "distribution": leaf_dist.tolist()}
        else:
            split_index = entmax15JAX(split_indices[node_id].T / temperature).T
            split_value = split_values[node_id]

            left_child_id = 2 * node_id + 1
            right_child_id = 2 * node_id + 2

            return {
                "type": "internal",
                "split_index": split_index,
                "split_value": split_value,
                "left_child": build_tree(left_child_id),
                "right_child": build_tree(right_child_id),
            }

    return build_tree(0)


def convert_to_child_representation(split_values, split_indices, leaf_values, features_by_estimator):
    num_internal_nodes = split_values.shape[0]
    num_leaf_nodes = leaf_values.shape[0]

    def build_tree(node_id):
        if node_id >= num_internal_nodes:
            leaf_index = node_id - num_internal_nodes
            leaf_dist = leaf_values[leaf_index]
            return {
                "type": "leaf",
                "action": leaf_dist,  # np.argmax(leaf_dist),
                "distribution": leaf_dist.tolist(),
            }
        else:
            split_index = np.argmax(split_indices[node_id])
            split_index = features_by_estimator[split_index]
            split_value = split_values[node_id, split_index]
            if np.round(split_value) == 1 and split_value < 1:
                split_value = 0.99
            elif np.round(split_value) == -1 and split_value > -1:
                split_value = -0.99

            left_child_id = 2 * node_id + 1
            right_child_id = 2 * node_id + 2

            return {
                "type": "internal",
                "split_index": int(split_index),
                "split_value": float(split_value),
                "left_child": build_tree(left_child_id),
                "right_child": build_tree(right_child_id),
            }

    return build_tree(0)


def convert_to_discrete_tree(params: Mapping, action_type: str, temperature: float = 1.0) -> Mapping[str, Any]:
    """
    Convert a trained soft decision tree (SDT) into a discrete decision tree.

    Args:
        params (dict): The parameters of the trained SDT.
        temperature (float): The temperature parameter for the entmax activation function.

    Returns:
        dict: The parameters of the discrete decision tree.
    """
    # Create a deep copy of the parameters to avoid modifying the original parameters
    new_params = unfreeze(copy.deepcopy(params))  # type: ignore

    beta = new_params["params"]["SDT_0"]["inner_nodes"]["layers_0"]["kernel"]
    beta = entmax15JAX(beta.T / temperature).T

    # print('beta', beta)
    phi = new_params["params"]["SDT_0"]["inner_nodes"]["layers_0"]["bias"]

    # Obtain the index of the feature to use
    j = jnp.argmax(beta, axis=0)

    one_hot_beta = jax.nn.one_hot(j, num_classes=beta.shape[0]).T

    # Normalize phi
    # print('beta', beta)
    # print('jnp.sum(beta * one_hot_beta, axis=0)', jnp.sum(beta * one_hot_beta, axis=0))
    normalized_phi = phi / jnp.sum(beta * one_hot_beta, axis=0)
    # print('one_hot_beta', one_hot_beta)
    # print('jnp.sum(beta * one_hot_beta, axis=-1)', jnp.sum(beta * one_hot_beta, axis=0))

    # Update params
    new_params["params"]["SDT_0"]["inner_nodes"]["layers_0"]["kernel"] = one_hot_beta
    new_params["params"]["SDT_0"]["inner_nodes"]["layers_0"]["bias"] = normalized_phi

    if action_type == "discrete":
        beta_leaf = new_params["params"]["SDT_0"]["leaf_nodes"]["kernel"]

        # Obtain the index of the feature to use
        j = jnp.argmax(beta_leaf, axis=1)

        # Create one-hot vector for beta
        one_hot_beta_leaf = jax.nn.one_hot(j, num_classes=beta_leaf.shape[1])

        # Update params
        new_params["params"]["SDT_0"]["leaf_nodes"]["kernel"] = one_hot_beta_leaf
    else:
        log_std = new_params["params"]["SDT_0"]["log_std"]

        new_params["params"]["SDT_0"]["log_std"] = jnp.zeros_like(log_std)
    return freeze(new_params)


def prune_and_merge_tree(
    node: dict, split_ranges: dict[Any, tuple[float, float]], constraints: Optional[dict] = None, continuous=False
):
    """
    Prune a decision tree based on predefined ranges for each split index, merge leaf nodes with the same distribution,
    and remove redundant paths that cannot be taken because previous splits already predetermine the path.

    Args:
    - node (dict): The decision tree node (root node initially).
    - split_ranges (dict): A dictionary where keys are split indices and values are tuples of (min_value, max_value).
    - constraints (dict): A dictionary to keep track of constraints on split indices.

    Returns:
    - dict: The pruned and merged decision tree or the subtree if the current node is pruned.
    """
    if constraints is None:
        constraints = {}

    if node["type"] == "leaf":
        return node

    split_index = node["split_index"]
    split_value = node["split_value"]

    if split_index in split_ranges:
        min_value, max_value = split_ranges[split_index]
        if split_value < min_value or split_value > max_value:
            # If the split value is outside the range, prune this node
            # Return left child if it exists, otherwise right child if it exists, else None
            if split_value < min_value:  # and node['right_child']['type'] != 'leaf':
                if node["right_child"]["type"] != "leaf":
                    return prune_and_merge_tree(node["right_child"], split_ranges, constraints, continuous=continuous)
                else:
                    return node["right_child"]
            elif split_value > max_value:  # and node['left_child']['type'] != 'leaf':
                if node["left_child"]["type"] != "leaf":
                    return prune_and_merge_tree(node["left_child"], split_ranges, constraints, continuous=continuous)
                else:
                    return node["left_child"]
            else:
                return None  # node['left_child'] if node['left_child'] else node['right_child']

    # Check if the current split is redundant based on constraints
    if split_index in constraints:
        min_constraint, max_constraint = constraints[split_index]
        if (split_value >= min_constraint) or (split_value <= max_constraint):
            # The split is redundant, remove this node and move its child up
            if split_value <= max_constraint:  # and node['right_child']['type'] != 'leaf':
                if node["right_child"]["type"] != "leaf":
                    return prune_and_merge_tree(node["right_child"], split_ranges, constraints, continuous=continuous)
                else:
                    return node["right_child"]
            elif split_value >= min_constraint:  # and node['left_child']['type'] != 'leaf':
                if node["left_child"]["type"] != "leaf":
                    return prune_and_merge_tree(node["left_child"], split_ranges, constraints, continuous=continuous)
                else:
                    return node["left_child"]
            else:
                return None  # node['left_child'] if node['left_child'] else node['right_child']

    # Update constraints based on the current split
    new_constraints_left = constraints.copy()
    new_constraints_right = constraints.copy()
    if split_index in new_constraints_left:
        new_constraints_left[split_index] = (
            min(split_value, new_constraints_left[split_index][0]),
            new_constraints_left[split_index][1],
        )
    else:
        new_constraints_left[split_index] = (split_value, -np.inf)
    if split_index in new_constraints_right:
        new_constraints_right[split_index] = (
            new_constraints_right[split_index][0],
            max(split_value, new_constraints_right[split_index][1]),
        )
    else:
        new_constraints_right[split_index] = (np.inf, split_value)

    # print(new_constraints_left, new_constraints_right)
    # Recursively prune and merge left and right children
    node["left_child"] = prune_and_merge_tree(
        node["left_child"], split_ranges, new_constraints_left, continuous=continuous
    )
    node["right_child"] = prune_and_merge_tree(
        node["right_child"], split_ranges, new_constraints_right, continuous=continuous
    )

    # If both children are leaves with the same distribution, merge them
    if (
        not continuous
        and node["left_child"]
        and node["left_child"]["type"] == "leaf"
        and node["right_child"]
        and node["right_child"]["type"] == "leaf"
        and
        # node['left_child']['distribution'] == node['right_child']['distribution']):
        np.argmax(node["left_child"]["distribution"]) == np.argmax(node["right_child"]["distribution"])
    ):
        return node["left_child"]

    # If both children are pruned, prune this node too
    if node["left_child"] is None and node["right_child"] is None:
        return None

    return node


def count_nodes(tree):
    if tree["type"] == "leaf":
        return {"internal": 0, "leaf": 1}

    left_counts = count_nodes(tree["left_child"])
    right_counts = count_nodes(tree["right_child"])

    return {
        "internal": 1 + left_counts["internal"] + right_counts["internal"],
        "leaf": left_counts["leaf"] + right_counts["leaf"],
    }
