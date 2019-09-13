import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn import tree


def get_decision_tree_boundaries(tree,
                                 feature_num,
                                 class_num,
                                 boundary_val=100):
    """
    Args:
        tree (sklearn.tree.DecisionTree):
        feature_num (int):
        class_num (int):

    The way this is written expects that the classes continuously count
    from zero
    """
    # grab tree features
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    value = tree.tree_.value
    classes = tree.classes_

    # calculate useful numbers
    node_num = children_left.shape[0]
    non_leaves = np.nonzero(children_left + 1)[0]

    lower_bounds = np.empty((node_num, feature_num))
    upper_bounds = np.empty((node_num, feature_num))
    values = np.empty((node_num, class_num))

    # subroutine to define boxes with depth first walk
    def dfs(idx=0,
            lower=[-boundary_val] * feature_num,
            upper=[boundary_val] * feature_num):
        """
        Args:
            idx:
            lower:
            upper:
        """
        # -1 is the magic number used by scikit-learn to signal a leaf
        if children_left[idx] != -1:
            old_upper_val = upper[feature[idx]]
            upper[feature[idx]] = threshold[idx]
            dfs(children_left[idx], lower, upper)
            upper[feature[idx]] = old_upper_val

            lower[feature[idx]] = threshold[idx]
            dfs(children_right[idx], lower, upper)
        else:
            vals = value[idx][0] / np.sum(value[idx])
            lower_bounds[idx] = lower
            upper_bounds[idx] = upper
            values[idx] = np.dot(vals, np.eye(class_num)[classes])

    # run and delete non-leaves
    dfs()
    lower_bounds = np.delete(lower_bounds, non_leaves, axis=0)
    upper_bounds = np.delete(upper_bounds, non_leaves, axis=0)
    values = np.delete(values, non_leaves, axis=0)

    return lower_bounds, upper_bounds, values


def transductive_box_inference(mu, sigma, lower_bounds, upper_bounds, values):
    """
    Args:
    - mu, sigma (Tensor): shape [batch_size, latent_dimension]
    - lower_bounds, upper_bounds (Tensor): shape [box_num, latent_dimension]
    - values (Tensor): shape [box_num, class_num] proportion of each class in the membership of the box
    """
    mu = tf.cast(mu, tf.float64)
    sigma = tf.cast(sigma, tf.float64)
    # broadcast mu, sigma, and bounds to
    # shape [batch_size, box_num, latent_dimension]
    mu = tf.tile(tf.expand_dims(mu, 1), [1, tf.shape(lower_bounds)[0], 1])
    sigma = tf.tile(
        tf.expand_dims(sigma, 1), [1, tf.shape(lower_bounds)[0], 1])
    lower_bounds = tf.tile(
        tf.expand_dims(lower_bounds, 0), [tf.shape(mu)[0], 1, 1])
    upper_bounds = tf.tile(
        tf.expand_dims(upper_bounds, 0), [tf.shape(mu)[0], 1, 1])

    # integral over CDF between bounds per dimension, rectifying for numerical error
    # in the tails of the CDF
    dist = tfp.distributions.Normal(mu, sigma, True, False)
    upper_bounds = tf.cast(upper_bounds, tf.double)
    dim_probs = tf.nn.relu(dist.cdf(upper_bounds) - dist.cdf(lower_bounds))

    # for each box, calculate probability that a sample falls in
    # as we assume the Gaussian has diagonal covariance, this is a product
    box_prob = tf.reduce_prod(dim_probs, axis=2)

    # finally, calculate joint probability of P(A and B_i) as above
    box_prob = tf.expand_dims(tf.transpose(box_prob), 2)
    values = tf.expand_dims(values, 1)
    y = tf.matmul(box_prob, values)
    # and marginalize out boxes choice by summing
    y = tf.reduce_sum(y, axis=0)
    return y
