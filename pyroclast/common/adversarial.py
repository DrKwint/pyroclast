import tensorflow as tf
import numpy as np


def fast_gradient_method(forward_fn, x, eps, norm):
    """
    Goodfellow et al.'s fast gradient adversarial attack method which assumes a
    neural network is mostly linear (i.e., activated with ReLUs or similar) and
    generates a perturbation with a single gradient step in data space.

    Args:
        forward_fn: forward pass from input data to a tensor of shape () per input datum
        x (Tensor): input data
        eps (float): norm constraint on the delta
        norm (1, 2, or np.inf): norm choosing constraint set to optimize

    Returns:
        perturbation (Tensor): data-space adversarial perturbation which, added \
            to the original input, minimizes the output of forward_fn
    """
    jacobian = compute_all_gradients(forward_fn, x)
    perturbation = linear_optimization(jacobian, eps, norm)
    return -perturbation


def compute_all_gradients(forward_fn, x):
    """
    Computes the gradient of the class logits w.r.t. input tensor

    Args:
        forward_fn: forward pass from input data to a tensor of shape () per input datum
        x (Tensor): input data

    Returns:
        jacobian (Tensor): containing the gradient of the output with respect to the input tensor.
    """
    with tf.GradientTape(persistent=True,
                         watch_accessed_variables=False) as tape:
        tape.watch(x)
        output = forward_fn(x)
    jacobian = tape.gradient(output, x)
    return jacobian


def linear_optimization(jacobian, eps, norm):
    """
    Optimizes for the perterbation which will most minimize the network output
    under the constraint that the norm of the perturbation is less than eps.

    Args:
        jacobian (Tensor): shape [N,...] with gradients of output w.r.t. data
        eps (float): norm constraint on the delta
        norm (1, 2, or np.inf): norm choosing constraint set to optimize

    Returns:
        scaled_perturbation (Tensor): perturbation of shape [N,...] which \
            minimizes network output under the given norm/eps upper bound
    """
    axis = list(set(range(len(jacobian.shape))) - set([0, 1, 2]))
    avoid_zero_div = 1e-12

    if norm == np.inf:
        optimal_perturbation = tf.sign(jacobian)
        optimal_perturbation = tf.stop_gradient(optimal_perturbation)
    elif norm == 1:
        abs_grad = tf.abs(jacobian)
        sign = tf.sign(jacobian)
        max_abs_grad = tf.reduce_max(abs_grad, axis, keepdims=True)
        tied_for_max = tf.dtypes.cast(tf.equal(abs_grad, max_abs_grad),
                                      dtype=tf.float32)
        num_ties = tf.reduce_sum(tied_for_max, axis, keepdims=True)
        optimal_perturbation = sign * tied_for_max / num_ties
    elif norm == 2:
        square = tf.maximum(
            avoid_zero_div,
            tf.reduce_sum(tf.square(jacobian), axis, keepdims=True))
        optimal_perturbation = jacobian / tf.sqrt(square)
    else:
        raise NotImplementedError()

    scaled_perturbation = tf.multiply(eps, optimal_perturbation)
    return scaled_perturbation
