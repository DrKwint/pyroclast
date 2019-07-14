import tensorflow as tf


def calculate_binary_gamma_tilde(label, hypothesis, prev_hypothesis,
                                 distribution):
    del distribution
    assert tf.abs(tf.reduce_sum(distribution, axis=1) - 1.) < 1e-5
    label = tf.cast(label, tf.float32)
    gamma_tilde = distribution * label * hypothesis
    prev_gamma_tilde = distribution * label * prev_hypothesis
    gamma = tf.sqrt((tf.square(gamma_tilde) - tf.square(prev_gamma_tilde)
                     ) / 1 - tf.square(prev_gamma_tilde))
    return gamma


def update_binary_distribution(weak_module_classifier, label,
                               binary_distribution):
    """Each of the classifiers should be a vector of alpha*o - alpha*o for a pair of o hypotheses"""
    weak_module_classifier = tf.squeeze(weak_module_classifier)
    binary_distribution = tf.squeeze(binary_distribution)
    numerator = binary_distribution * tf.exp(
        -1 * tf.cast(label, tf.float32) * weak_module_classifier)
    return numerator / tf.reduce_sum(numerator)


def binary_loss(weak_module_classifier, label, distribution):
    weak_module_classifier = tf.squeeze(weak_module_classifier)
    return tf.reduce_mean(distribution * tf.exp(
        tf.cast(label, tf.float32) * weak_module_classifier))


def calculate_multiclass_gamma_tilde(label, hypothesis, prev_hypothesis,
                                     cost_fn_values, boosted_classifier):
    del prev_hypothesis
    numerator = -1 * tf.reduce_sum(cost_fn_values * boosted_classifier)

    batch_size = tf.cast(tf.shape(label)[0], tf.int64)
    label_idxs = tf.stack([tf.range(batch_size), label], axis=-1)
    label_cost_fn_vals = tf.gather_nd(cost_fn_values, label_idxs)
    denominator = tf.reduce_sum(
        tf.reduce_sum(cost_fn_values, axis=1) - label_cost_fn_vals)
    return numerator / denominator


def initial_multiclass_distribution(label, num_classes):
    batch_size = tf.cast(tf.shape(label)[0], tf.int64)
    return (tf.ones([batch_size, num_classes]) -
            num_classes * tf.one_hot(label, num_classes)) / num_classes


class UpdateMulticlassDistribution(object):
    """Needs to keep s_t state info in order to keep __call__ equiv to the binary case"""

    def __init__(self):
        self.state = 0.

    def __call__(self, *argv, **kwargs):
        return self.update_multiclass_distribution(*argv, **kwargs)

    def update_multiclass_distribution(self, weak_module_classifier, label,
                                       multiclass_distribution):
        self.state += weak_module_classifier

        # get value at correct label index
        batch_size = tf.cast(tf.shape(label)[0], tf.int64)
        label_num = self.state.shape[1]
        label_idxs = tf.stack([tf.range(batch_size), label], axis=-1)
        label_vals = tf.gather_nd(self.state, label_idxs)
        tiled_label_vals = tf.tile(
            tf.expand_dims(label_vals, 1), [1, label_num])

        incorrect_idxs = tf.exp(
            self.state -
            tiled_label_vals)  # this leaves 1 at the correct indices
        correct_idxs = -1 * tf.reduce_sum(
            tf.exp(self.state - tiled_label_vals),
            axis=1)  # exp has 1 at correct indices so each sum has extra -1
        if tf.math.is_nan(tf.reduce_mean(self.state)):
            print("NaN in update_multiclass_distribution")
            exit()
        return incorrect_idxs + tf.scatter_nd(
            label_idxs, correct_idxs,
            [batch_size, label_num])  # sum cancels ones


def multiclass_loss(weak_module_classifier, label, state):
    """Calculates loss (without state)
        Args:
            weak_module_classifier: h
            label: 1D int labels
            distribution: same shape as h
        """
    # Extract vector of logits in the correct class over the batch
    batch_size = tf.cast(tf.shape(label)[0], tf.int64)
    label_num = weak_module_classifier.shape[1]
    label_idxs = tf.stack([tf.range(batch_size), label], axis=-1)

    label_prediction_vals = tf.gather_nd(weak_module_classifier, label_idxs)
    label_prediction_vals = tf.tile(
        tf.expand_dims(label_prediction_vals, 1), [1, label_num])
    try:
        state_label_vals = tf.gather_nd(state, label_idxs)
        state_label_vals = tf.tile(
            tf.expand_dims(state_label_vals, 1), [1, label_num])
    except:
        state_label_vals = tf.zeros_like(label_prediction_vals)

    h_term = weak_module_classifier - label_prediction_vals  # where y_i == l, there is a 0
    s_term = state - state_label_vals  # where y_i == l, there is a 0
    loss = tf.reduce_sum(
        tf.exp(h_term + s_term),
        axis=1) - 1  # subtract 1 to handle the product of the y_i == l index
    return loss
