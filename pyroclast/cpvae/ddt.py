import os

import numpy as np
import sklearn
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier

tfd = tfp.distributions


class DDT(tf.Module):
    """Differentiable decision tree which classifies on the parameters of a Gaussian"""

    def __init__(self, max_depth):
        self.decision_tree = DecisionTreeClassifier(max_depth=max_depth)

    def classify_analytic(self, loc, scale_diag):
        return transductive_box_inference(loc, scale_diag, self.dims,
                                          self.threshold, self.leaf_class_prob,
                                          self.r_mask)

    def classify_numerical(self, z_posterior):
        raise Exception()

    def update_model_tree(self, ds, posterior_fn, oversample, debug):
        repeated_ds = ds.repeat(oversample)
        if debug:
            repeated_ds = tqdm(repeated_ds)
        # calculate latent variable values and labels
        labels, z_samples = zip(*[(
            batch['label'],
            posterior_fn(tf.dtypes.cast(batch['image'], tf.float32)).sample())
                                  for batch in repeated_ds])
        labels = np.concatenate(labels).astype(np.int32)
        z_samples = np.concatenate(z_samples)

        # train decision tree
        self.decision_tree.fit(z_samples, labels)
        score = self.decision_tree.score(z_samples, labels)
        self.dims, self.threshold, self.leaf_class_prob, self.r_mask = get_decision_tree_boundaries(
            self.decision_tree)
        return score

    def save_dot(self, output_dir, epoch):
        sklearn.tree.export_graphviz(self.decision_tree,
                                     out_file=os.path.join(
                                         output_dir,
                                         'ddt_epoch{}.dot'.format(epoch)),
                                     filled=True,
                                     rounded=True)


def get_decision_tree_boundaries(dtree):
    # grab tree features
    children_left = dtree.tree_.children_left
    children_right = dtree.tree_.children_right
    feature = dtree.tree_.feature
    threshold = dtree.tree_.threshold
    value = dtree.tree_.value

    # calculate useful numbers
    leaves = np.where(children_left == -1)[0]
    leaf_values = tf.squeeze(value[leaves])
    norm_leaf_values = leaf_values / tf.expand_dims(
        tf.reduce_sum(leaf_values, axis=-1), -1)

    depth = int(np.ceil(np.log2(len(leaves))))
    dim = np.zeros([depth, len(leaves)], dtype=np.int32)
    split = np.zeros([depth, len(leaves)])
    r_mask = np.zeros([depth, len(leaves)], dtype=np.int32)

    def dfs(idx=0, depth=0, lower_range=0, upper_range=len(leaves)):
        if children_left[idx] != -1:
            # nodes only have a split if they have children
            split[depth, lower_range:upper_range] = threshold[idx]
            dim[depth, lower_range:upper_range] = feature[idx]
            r_mask[depth, (lower_range + upper_range) // 2:upper_range] = 1
            dfs(children_left[idx], depth + 1, lower_range,
                (lower_range + upper_range) // 2)
        if children_right[idx] != -1:
            dfs(children_right[idx], depth + 1,
                (lower_range + upper_range) // 2, upper_range)

    dfs()

    return dim, split, norm_leaf_values, r_mask


def transductive_box_inference(loc, scale_diag, dim, split, values, r_mask):
    built_mu = tf.transpose(
        tf.gather_nd(tf.transpose(loc), tf.expand_dims(dim, -1)), [2, 0, 1])
    built_sigma = tf.transpose(
        tf.gather_nd(tf.transpose(scale_diag), tf.expand_dims(dim, -1)),
        [2, 0, 1])
    dist = tfd.Normal(tf.cast(built_mu, tf.float64),
                      tf.cast(built_sigma, tf.float64))
    left_log_probs = dist.log_cdf(tf.cast(split, tf.float64))
    right_log_probs = dist.log_survival_function(tf.cast(split, tf.float64))
    log_probs = (1 - r_mask) * left_log_probs + (r_mask) * right_log_probs
    box_probs = tf.exp(tf.reduce_sum(log_probs, axis=1))
    class_probs = tf.matmul(box_probs, values)
    return class_probs
