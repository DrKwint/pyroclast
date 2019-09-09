import tensorflow as tf


def build_cost_matrix(num_classes, labels):
    """
    Args:
        num_classes (int):
        labels (Tensor): dtype int, shape (dataset_size,)

    Returns
        A matrix with rows indexed by data index and columns indexed by label
    """
    dataset_size = labels.shape.as_list()[0]
    ones_matrix = tf.ones([dataset_size, num_classes])
    adjustment_matrix = tf.eye(num_classes) * tf.one_hot(
        labels, num_classes) * num_classes
    return ones_matrix - adjustment_matrix


def build_cost_matrix_update(cost_matrix):
    pass
