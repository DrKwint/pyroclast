import tensorflow as tf

from pyroclast.prototype.tf_util import l2_convolution


class PrototypeLayer(tf.Module):

    def __init__(self, num_prototypes, prototype_dim):
        """
        Args:
            num_prototypes (int): Number of prototype vectors in the model
            prototype_dim (int): Dimensionality of each prototype vector
        """
        self.max_pool = tf.keras.layers.GlobalMaxPool2D()

    def __call__(self, z, prototypes, epsilon=1e-4):
        """
        Args:
            z (tf.Tensor): shape [N_1,7,7,C]
            prototypes (tf.Tensor): shape [N_2,C]
            epsilon (float): Optional, some sufficiently small value
        """

        distances = l2_convolution(z, prototypes)
        similarities = tf.math.log((distances + 1) / (distances + epsilon))

        return distances, similarities
