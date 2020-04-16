import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sklearn

from pyroclast.cpvae.util import ensure_dir_exists


class DDT(tf.Module):
    """Differentiable decision tree which classifies on the parameters of a Gaussian"""

    def __init__(self, decision_tree, num_classes):
        self.decision_tree = decision_tree
        self.num_classes = num_classes
        self.lower = None
        self.upper = None
        self.values = None

    def __call__(self, loc, scale_diag):
        return transductive_box_inference(loc, scale_diag, self.lower,
                                          self.upper, self.values)

    def update_model_tree(self, ds, posterior_fn):
        # calculate latent variable values and labels
        labels, z_vals = zip(*[(
            batch['label'],
            posterior_fn(tf.dtypes.cast(batch['image'], tf.float32)).sample())
                               for batch in ds])
        labels = np.concatenate(labels).astype(np.int32)
        z_vals = np.concatenate(z_vals)

        # train decision tree
        self.decision_tree.fit(z_vals, labels)
        self.lower, self.upper, self.values = get_decision_tree_boundaries(
            self.decision_tree, z_vals.shape[-1], self.num_classes)

    def save_dot(self, output_dir, epoch):
        ensure_dir_exists(output_dir)
        sklearn.tree.export_graphviz(self.decision_tree,
                                     out_file=os.path.join(
                                         output_dir,
                                         'ddt_epoch{}.dot'.format(epoch)),
                                     filled=True,
                                     rounded=True)


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


def transductive_box_inference(mu, sigma, lower_bounds, upper_bounds,
                               conditional_class_prob):
    """
    Calculates the probability that a sample of a multivariate Gaussian, defined :math:`N(\mu, \sigma I)`, will be classified into class c

    Args:
        mu (Tensor): shape [batch_size, latent_dimension]
        sigma (Tensor): identical shape and dtype to `mu`
        lower_bounds (Tensor): shape [num_boxes, latent_dimension], defines the lower and upper bounds of each box in each dimension
        upper_bounds (Tensor): identical shape and dtype to `lower_bounds`
        conditional_class_prob (Tensor): shape [num_boxes, class_num] weight of each class per box

    Returns:
        Tensor of shape [batch_size, class_num] with the parameters of a discrete distribution per datum
    """
    # Infer dimensions and make sure data types are set as needed
    batch_size = tf.shape(mu)[0]
    num_boxes = tf.shape(lower_bounds)[0]
    mu = tf.cast(mu, tf.float64)
    sigma = tf.cast(sigma, tf.float64)
    upper_bounds = tf.cast(upper_bounds, tf.double)
    lower_bounds = tf.cast(lower_bounds, tf.double)

    # broadcast mu, sigma, and bounds to
    # shape [batch_size, num_boxes, latent_dimension]
    mu = tf.tile(tf.expand_dims(mu, 1), [1, num_boxes, 1])
    sigma = tf.tile(tf.expand_dims(sigma, 1), [1, num_boxes, 1])
    lower_bounds = tf.tile(tf.expand_dims(lower_bounds, 0), [batch_size, 1, 1])
    upper_bounds = tf.tile(tf.expand_dims(upper_bounds, 0), [batch_size, 1, 1])

    # integral over PDF between bounds per dimension, rectify numerical error
    dist = tfp.distributions.Normal(mu,
                                    sigma,
                                    validate_args=True,
                                    allow_nan_stats=False)
    dim_log_probs = tf.math.log(
        1e-10 + tf.nn.relu(dist.cdf(upper_bounds) - dist.cdf(lower_bounds)))

    # For each box, calculate probability that a sample falls in
    # We assume the Gaussian has diagonal covariance so this is a product
    box_prob = tf.math.exp(tf.reduce_sum(dim_log_probs, axis=2))

    # finally, calculate joint probability of P(A and B_i) as above
    box_prob = tf.expand_dims(tf.transpose(box_prob), 2)
    conditional_class_prob = tf.expand_dims(conditional_class_prob, 1)
    joint_prob = tf.matmul(box_prob, conditional_class_prob)
    # and marginalize out boxes choice by summing followed by normalization
    class_prob = tf.reduce_sum(joint_prob, axis=0)
    class_prob = class_prob / tf.expand_dims(tf.reduce_sum(class_prob, axis=1),
                                             1)
    return class_prob
