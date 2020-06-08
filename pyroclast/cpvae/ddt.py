import os
import sonnet as snt

import numpy as np
import sklearn
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier

tfd = tfp.distributions


def compute_jitter_cholesky(x):
    print(x.shape)
    try:
        cholesky = tf.linalg.cholesky(x)
        return cholesky
    except Exception:
        jitter = 1e-15
        while jitter < 1.0:
            try:
                cholesky = tf.linalg.cholesky(
                    x + tf.linalg.diag(jitter * tf.ones([x.shape[0]])))
                return cholesky
            except Exception:
                jitter = jitter * 10
        raise Exception("Chlolesky impossible")


class DDT(tf.Module):
    """Differentiable decision tree which classifies on the parameters of a Gaussian"""

    def __init__(self, max_depth, use_analytic):
        self.decision_tree = DecisionTreeClassifier(max_depth=max_depth)
        self.use_analytic = use_analytic

    def __call__(self, z_posterior):
        if self.use_analytic:
            leaf_probs, y_hat = self.classify_analytic(
                z_posterior.parameters['loc'],
                z_posterior.parameters['scale_diag'])
        else:
            leaf_probs, y_hat = self.classify_numerical(z_posterior)
        return y_hat

    def classify_analytic(self, loc, scale_diag):
        built_mu = tf.transpose(
            tf.gather_nd(tf.transpose(loc), tf.expand_dims(self.dims, -1)),
            [2, 0, 1])
        built_sigma = tf.transpose(
            tf.gather_nd(tf.transpose(scale_diag),
                         tf.expand_dims(self.dims, -1)), [2, 0, 1])
        dist = tfd.Normal(tf.cast(built_mu, tf.float64),
                          tf.cast(built_sigma, tf.float64))
        leaf_probs = calculate_leaf_probs(dist, self.threshold, self.r_mask)
        class_probs = calculate_class_likelihood(leaf_probs,
                                                 self.leaf_class_prob)
        return tfd.Categorical(leaf_probs), tfd.Categorical(
            tf.cast(class_probs, tf.float32))

    def classify_numerical(self, z_posterior, num_samples=100):
        """
        Using logP(l | z) = logP(z | l) + logP(l) - logp(z)
        """
        z = tf.cast(z_posterior.sample(num_samples), tf.float64)
        z_l = tf.convert_to_tensor([
            self.tree_distribution.cat.probs_parameter()[i] * c.prob(z)
            for i, c in enumerate(self.tree_distribution.components)
        ])
        print([c for c in self.tree_distribution.components])

        # mean over sample dim, transpose to get batch out front
        sum_z_l = tf.reduce_sum(z_l, 0)
        sample_l_z = z_l / (sum_z_l + 1e-127)
        l_z = tf.reduce_mean(sample_l_z, 1)
        l_z = tf.transpose(l_z)
        return tfd.Categorical(l_z), calculate_class_likelihood(
            l_z, self.leaf_class_prob)

    def update_model_tree(self, ds, posterior_fn, oversample, debug):
        repeated_ds = ds.repeat(oversample)
        if debug:
            repeated_ds = tqdm(repeated_ds)
        # calculate latent variable values and labels
        labels, z_samples = zip(*[(
            batch['label'],
            posterior_fn(tf.dtypes.cast(batch['image'], tf.float32) /
                         255.).sample()) for batch in repeated_ds])
        labels = np.concatenate(labels).astype(np.int32)
        z_samples = np.concatenate(z_samples)
        batch_size = z_samples.shape[0]
        flat_z_samples = np.reshape(z_samples, [batch_size, -1])

        # train decision tree
        self.decision_tree.fit(flat_z_samples, labels)
        score = self.decision_tree.score(flat_z_samples, labels)
        self.dims, self.threshold, self.leaf_class_prob, self.r_mask = get_decision_tree_boundaries(
            self.decision_tree)
        self.tree_distribution = self.learn_leaf_distributions(z_samples)
        return score

    def learn_leaf_distributions(self, data):
        num_data = data.shape[0]
        flat_data = np.reshape(data, [num_data, -1])
        children_left = self.decision_tree.tree_.children_left
        leaves = np.where(children_left == -1)[0]

        node_indicator = np.transpose(
            self.decision_tree.decision_path(flat_data))
        node_samples = lambda node_id: np.nonzero(node_indicator[node_id])[1]
        categorial_weights = []
        distributions = []
        for l in leaves:
            leaf_data = data[node_samples(l)]
            loc = tf.reduce_mean(leaf_data, 0)
            std = tf.reduce_mean(leaf_data - loc, 0)
            leaf_dist = tfd.MultivariateNormalDiag(loc=tf.cast(loc, tf.float64),
                                                   scale_diag=tf.cast(
                                                       std, tf.float64))
            if len(loc.shape) > 2:
                leaf_dist = tfd.Independent(leaf_dist,
                                            len(leaf_dist.batch_shape))
            distributions.append(leaf_dist)
            categorial_weights.append(leaf_data.shape[0] / num_data)
        print(distributions[0])
        print(tfd.Categorical(categorial_weights))
        return tfd.Mixture(cat=tfd.Categorical(
            tf.cast(categorial_weights, tf.float64)),
                           components=distributions)

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
    r_mask = np.zeros([depth, len(leaves)], dtype=np.float64)

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


def calculate_leaf_probs(dist, split, r_mask):
    left_log_probs = dist.log_cdf(tf.cast(split, tf.float64))
    right_log_probs = dist.log_survival_function(tf.cast(split, tf.float64))
    log_probs = (1 - r_mask) * left_log_probs + (r_mask) * right_log_probs
    leaf_probs = tf.exp(tf.reduce_sum(log_probs, axis=-2))
    return leaf_probs


def calculate_class_likelihood(leaf_probs, leaf_values):
    print(tf.matmul(tf.cast(leaf_probs, tf.float64), leaf_values).shape)
    return tf.matmul(tf.cast(leaf_probs, tf.float64), leaf_values)
