import tensorflow as tf
import numpy as np


def fast_gradient_sign_method(feature_fn, classify_fn, x, y, eps, norm):
    """
    Goodfellow et al.'s fast gradient adversarial attack method which assumes a
    neural network is mostly linear (i.e., activated with ReLUs or similar) and
    generates a perturbation with a single gradient step in data space.

    Args:
        feature_fn (x -> z): where x has shape [NHWCh]
        classify_fn (z -> y): where y has shape [NCl] and z identical to feature_fn
        x (Tensor): data, assumed to be [NHWCh]
        y (Tensor): labels, assumed int typed
        eps (float): norm constraint on the delta
        norm (1, 2, or np.inf): norm choosing constraint set to optimize

    Returns:
        perturbation (Tensor): adversarial perturbation which minimizes \
            feature/class outer product with shape [N,F,Cl,H,W,Ch] where each \
            datum (H,W,Ch) is optimized against each feature/class pair (F,Cl)
    """
    jacobian = compute_all_gradients(feature_fn, classify_fn, x, y)
    perturbation = linear_optimization(jacobian, eps, norm)
    return perturbation


def compute_all_gradients(feature_fn, classify_fn, x, y):
    """
    Computes the gradient of the class logits w.r.t. input tensor

    Args:
        feature_fn (x -> z): where x has shape [NHWCh]
        classify_fn (z -> y): where y has shape [NCl] and z identical to feature_fn
        x (Tensor): data, assumed to be [NHWCh]
        y (Tensor): labels, assumed int typed

    Returns:
        jacobian (Tensor): containing the gradient of the feature/class outer \
            product with respect to the input tensor.
    """
    with tf.GradientTape(persistent=True,
                         watch_accessed_variables=False) as tape:
        tape.watch(x)
        features = feature_fn(x)
        fc_dot = tf.einsum('ij,ik->ijk', features, y)
    jacobian = tape.batch_jacobian(fc_dot, x)
    return jacobian


def linear_optimization(jacobian, eps, norm):
    """
    Optimizes for the perterbation which will most minimize the network output
    under the constraint that the norm of the perturbation is less than eps.

    Args:
        jacobian (Tensor): shape [N,F,Cl,...] with gradients of feature/class outer product w.r.t. data
        eps (float): norm constraint on the delta
        norm (1, 2, or np.inf): norm choosing constraint set to optimize

    Returns:
        scaled_perturbation (Tensor): perturbation of shape [N,F,Cl,...] which \
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
