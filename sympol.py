from typing import TYPE_CHECKING, Mapping

# from torch.autograd import Function
import chex
import jax
import jax.numpy as jnp
from flax import struct

from utils.jax_math import entmax15JAX


@struct.dataclass(kw_only=True, frozen=False)
class SYMPOL_RL:
    obs_dim: int = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    depth: int = struct.field(pytree_node=False)
    n_estimators: int = struct.field(pytree_node=False)
    action_type: str = struct.field(pytree_node=False)

    subset_fraction: float = 0.8

    # **kwargs is only used here to be compatible with the flax init procedure, i.e.
    # params = model.init(key, sample_input)
    def init(self, random_key, *args) -> dict[str, chex.Array]:
        estimator_weights_key, split_values_key, split_index_array_key, leaf_classes_array_key, logstd_key = (
            jax.random.split(random_key, 5)
        )

        internal_node_num = 2**self.depth - 1
        leaf_node_num = 2**self.depth

        leaf_classes_array_shape = (
            [self.n_estimators, leaf_node_num, self.action_dim]
            if self.action_type == "continuous"
            else [self.n_estimators, leaf_node_num, self.action_dim]
        )

        if self.n_estimators > 1:
            selected_variables = int(self.obs_dim * self.subset_fraction)
            selected_variables = min(selected_variables, 50)
            selected_variables = max(selected_variables, 10)
            selected_variables = min(selected_variables, self.obs_dim)
            if not selected_variables * self.n_estimators > 3 * self.obs_dim:
                selected_variables = self.obs_dim
        else:
            selected_variables = self.obs_dim
        mu, std = 0.0, 0.05
        estimator_weights = mu + std * jax.random.normal(
            key=estimator_weights_key,
            shape=[self.n_estimators, leaf_node_num],
            dtype=jnp.float32,
        )
        split_values = mu + std * jax.random.normal(
            key=split_values_key,
            shape=[self.n_estimators, internal_node_num, selected_variables],
            dtype=jnp.float32,
        )
        split_index_array = mu + std * jax.random.normal(
            key=split_index_array_key,
            shape=[self.n_estimators, internal_node_num, selected_variables],
            dtype=jnp.float32,
        )
        leaf_classes_array = mu + std * jax.random.normal(
            key=leaf_classes_array_key, shape=leaf_classes_array_shape, dtype=jnp.float32
        )

        log_std = mu + std * jax.random.normal(
            key=logstd_key,
            shape=[
                self.action_dim,
            ],
            dtype=jnp.float32,
        )

        params: dict[str, chex.Array] = {
            "estimator_weights": estimator_weights,
            "split_values": split_values,
            "split_idx_array": split_index_array,
            "leaf_array": leaf_classes_array,
            "log_std": log_std,
        }
        return params

    def init_indices(self, random_key) -> dict[str, chex.Array]:
        leaf_node_num = 2**self.depth
        if self.n_estimators > 1:
            selected_variables = int(self.obs_dim * self.subset_fraction)
            selected_variables = min(selected_variables, 50)
            selected_variables = max(selected_variables, 10)
            selected_variables = min(selected_variables, self.obs_dim)
            if not selected_variables * self.n_estimators > 3 * self.obs_dim:
                selected_variables = self.obs_dim
        else:
            selected_variables = self.obs_dim

        features_by_estimator = jnp.stack(
            [
                jax.random.choice(random_key + i, self.obs_dim, shape=(selected_variables,), replace=False, p=None)
                for i in range(self.n_estimators)
            ]
        )

        path_identifier_list = []
        internal_node_index_list = []
        for leaf_index in range(leaf_node_num):
            for current_depth in range(1, self.depth + 1):
                path_identifier = jnp.floor(leaf_index / (2 ** (self.depth - current_depth))) % 2
                internal_node_index = (
                    2 ** (current_depth - 1) + jnp.floor(leaf_index / (2 ** (self.depth - (current_depth - 1)))) - 1
                ).astype(jnp.int32)
                path_identifier_list.append(path_identifier)
                internal_node_index_list.append(internal_node_index)

        path_identifier_list = jnp.reshape(jnp.array(path_identifier_list, dtype=jnp.float32), (-1, self.depth))
        internal_node_index_list = jnp.reshape(jnp.array(internal_node_index_list, dtype=jnp.int32), (-1, self.depth))

        # jax.debug.print("path_identifier_list: {}", path_identifier_list)
        # jax.debug.print("internal_node_index_list: {}", internal_node_index_list)

        indices: dict[str, chex.Array] = {
            "features_by_estimator": features_by_estimator,
            "path_identifier_list": path_identifier_list,
            "internal_node_index_list": internal_node_index_list,
        }

        return indices

    @jax.jit
    def apply(self, params: Mapping, inputs: jax.Array, indices: dict):
        split_values = params["split_values"]
        estimator_weights = params["estimator_weights"]
        split_index_array = params["split_idx_array"]
        leaf_classes_array = params["leaf_array"]
        log_std = params["log_std"]

        features_by_estimator = indices["features_by_estimator"]
        path_identifier_list = indices["path_identifier_list"]
        internal_node_index_list = indices["internal_node_index_list"]

        # einsum syntax:
        #       - b is the batch size
        #       - e is the number of estimators
        #       - l the number of leaf nodes  (i.e. the number of paths)
        #       - i is the number of internal nodes
        #       - d is the depth (i.e. the length of each path)
        #       - n is the number of variables (one value is stored for each variable)

        X_estimator = inputs[:, features_by_estimator]

        # entmax transformaton
        split_index_array = entmax15JAX(split_index_array)

        # use ST-Operator to get one-hot encoded vector for feature index
        adjust_constant = split_index_array - jax.nn.one_hot(
            jnp.argmax(split_index_array, axis=-1), num_classes=split_index_array.shape[-1]
        )
        split_index_array = split_index_array - jax.lax.stop_gradient(adjust_constant)
        # jax.debug.print("split_index_array: {}", split_index_array)
        # as split_index_array_selected is one-hot-encoded, taking the sum over the last axis after multiplication results in selecting the desired value at the index
        s1_sum = jnp.einsum("ein,ein->ei", split_values, split_index_array)
        s2_sum = jnp.einsum("ben,ein->bei", X_estimator, split_index_array)
        # s2_sum = jnp.einsum("bn,ein->bei", inputs, split_index_array)

        # calculate the split (output shape: (b, e, i))
        node_result = (jax.nn.soft_sign(s1_sum - s2_sum) + 1) / 2
        adjust_constant = node_result - jnp.round(node_result)

        # use round operation with ST operator to get hard decision for each node
        node_result_corrected = node_result - jax.lax.stop_gradient(adjust_constant)

        # the resulting shape of the tensors is (b, e, l, d):
        node_result_extended = node_result_corrected[:, :, internal_node_index_list]
        # jax.debug.print("node_result_extended {}: {}", node_result_extended.shape, node_result_extended)
        # reduce the path via multiplication to get result for each path (in each estimator) based on the results of the corresponding internal nodes (output shape: (b, e, l))
        p = jnp.prod(
            ((1 - path_identifier_list) * node_result_extended + path_identifier_list * (1 - node_result_extended)),
            axis=3,
        )
        # jax.debug.print("p {}: {}", p.shape, p)
        # calculate instance-wise leaf weights for each estimator by selecting the weight of the selected path for each estimator
        estimator_weights_leaf = jnp.einsum("el,bel->be", estimator_weights, p)

        # use softmax over weights for each instance
        estimator_weights_leaf_softmax = jax.nn.softmax(estimator_weights_leaf)
        # jax.debug.print("estimator_weights_leaf_softmax {}: {}", estimator_weights_leaf_softmax.shape, estimator_weights_leaf_softmax)
        # get raw prediction for each estimator
        if self.action_type == "continuous":
            layer_output = jnp.einsum("elc,bel->bec", leaf_classes_array, p)
            layer_output = jnp.einsum("be,bec->bc", estimator_weights_leaf_softmax, layer_output)
            result = [layer_output, log_std]
        elif self.action_type == "discrete":
            layer_output = jnp.einsum("elc,bel->bec", leaf_classes_array, p)
            result = jnp.einsum("be,bec->bc", estimator_weights_leaf_softmax, layer_output)
        else:
            raise ValueError(f"Invalid action type {self.action_type}")

        return result


if TYPE_CHECKING:
    # Test assignability
    from ray_utilities.jax.jax_model import PureJaxModelProtocol

    _: type[PureJaxModelProtocol] = SYMPOL_RL
