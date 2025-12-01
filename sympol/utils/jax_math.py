"""
Taken from: https://github.com/deep-spin/entmax/blob/master/entmax/activations.py

An implementation of entmax (Peters et al., 2019). See
https://arxiv.org/pdf/1905.05702 for detailed description.

This builds on previous work with sparsemax (Martins & Astudillo, 2016).
See https://arxiv.org/pdf/1602.02068.
"""

# Author: Ben Peters
# Author: Vlad Niculae <vlad@vene.ro>
# License: MIT

import jax
import jax.numpy as jnp


def top_k_over_axisJAX(inputs, k, axis=-1, **kwargs):
    with jax.named_scope("top_k_along_axis"):
        if axis == -1:
            return jax.lax.top_k(inputs, k)

        perm_order = list(range(inputs.shape.ndims))
        perm_order.append(perm_order.pop(axis))
        inv_order = [perm_order.index(i) for i in range(len(perm_order))]

        input_perm = jnp.transpose(inputs, perm_order)
        input_perm_sorted, sort_indices_perm = jax.lax.top_k(input_perm, k=k, **kwargs)

        input_sorted = jnp.transpose(input_perm_sorted, inv_order)
        sort_indices = jnp.transpose(sort_indices_perm, inv_order)
    return input_sorted, sort_indices


def _make_ix_like(inputs, axis=-1):
    """creates indices 0, ... , input[axis] unsqueezed to input dimensios"""
    assert jnp.ndim(inputs) is not None
    rho = jnp.arange(1, inputs.shape[axis] + 1, dtype=jnp.float32)
    view = [1] * jnp.ndim(inputs)
    view[axis] = -1
    return jnp.reshape(rho, view)


def jax_gather_nd(params, indices):
    tuple_indices = tuple(indices[..., i] for i in range(indices.shape[-1]))
    return params[tuple_indices]


def gather_over_axisJAX(values, indices, gather_axis):
    assert jnp.ndim(indices) is not None
    assert jnp.ndim(indices) == jnp.ndim(values)

    ndims = jnp.ndim(indices)
    gather_axis = gather_axis % ndims
    shape = jnp.shape(indices)

    selectors = []
    for axis_i in range(ndims):
        if axis_i == gather_axis:
            selectors.append(indices)
        else:
            index_i = jnp.arange(shape[axis_i])
            index_i = jnp.reshape(index_i, [-1 if i == axis_i else 1 for i in range(ndims)])
            index_i = jnp.tile(index_i, [shape[i] if i != axis_i else 1 for i in range(ndims)])
            selectors.append(index_i)
    return jax_gather_nd(values, jnp.stack(selectors, axis=-1))


def entmax_threshold_and_supportJAX(inputs, axis=-1):
    """
    Computes clipping threshold for entmax1.5 over specified axis
    NOTE this implementation uses the same heuristic as
    the original code: https://tinyurl.com/pytorch-entmax-line-203
    :param inputs: (entmax1.5 inputs - max) / 2
    :param axis: entmax1.5 outputs will sum to 1 over this axis
    """
    with jax.named_scope("entmax_threshold_and_supportJAX"):
        num_outcomes = inputs.shape[axis]

        inputs_sorted, _ = top_k_over_axisJAX(inputs, k=num_outcomes, axis=axis, sorted=True)

        rho = _make_ix_like(inputs, axis=axis)

        mean = jnp.cumsum(inputs_sorted, axis=axis) / rho

        mean_sq = jnp.cumsum(jnp.square(inputs_sorted), axis=axis) / rho
        delta = (1 - rho * (mean_sq - jnp.square(mean))) / rho

        delta_nz = jax.nn.relu(delta)
        tau = mean - jnp.sqrt(delta_nz)

        support_size = jnp.sum(jnp.less_equal(tau, inputs_sorted), axis=axis, keepdims=True)

        tau_star = gather_over_axisJAX(tau, support_size - 1, axis)
    return tau_star, support_size


def entmax15JAX(inputs, axis=-1) -> jnp.ndarray:
    # Implementation taken from: https://github.com/deep-spin/entmax/tree/master/entmax
    """
    Entmax 1.5 implementation, heavily inspired by
     * paper: https://arxiv.org/pdf/1905.05702.pdf
     * pytorch code: https://github.com/deep-spin/entmax
    :param inputs: similar to softmax logits, but for entmax1.5
    :param axis: entmax1.5 outputs will sum to 1 over this axis
    :return: entmax activations of same shape as inputs
    """

    @jax.custom_gradient
    def _entmax_inner(inputs):
        with jax.named_scope("entmax"):
            inputs = inputs / 2  # divide by 2 so as to solve actual entmax
            inputs -= jnp.max(inputs, axis, keepdims=True)  # subtract max for stability

            threshold, _ = entmax_threshold_and_supportJAX(inputs, axis)
            outputs_sqrt = jax.nn.relu(inputs - threshold)
            outputs = jnp.square(outputs_sqrt)

        def grad_fn(d_outputs):
            with jax.named_scope("entmax_grad"):
                d_inputs = d_outputs * outputs_sqrt
                q = jnp.sum(d_inputs, axis=axis, keepdims=True)
                q = q / jnp.sum(outputs_sqrt, axis=axis, keepdims=True)
                d_inputs -= q * outputs_sqrt
                return d_inputs

        return outputs, grad_fn

    return _entmax_inner(inputs)  # type: ignore[return-type]  # see custom_gradient wrapper
