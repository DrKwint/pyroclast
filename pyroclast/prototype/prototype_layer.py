import tensorflow as tf


class PrototypeLayer(tf.Module):

    def __init__(self, num_prototypes, prototype_dim):
        self.prototype_dim = prototype_dim
        self.num_prototypes = num_prototypes
        # using ones here is a temporary measure until we figure out a better init
        self.prototypes = tf.Variable(tf.ones([num_prototypes, prototype_dim]),
                                      trainable=False)
        self.max_pool = tf.keras.layers.GlobalMaxPool2D()

    def __call__(self, z, epsilon=1e-4):
        """
        Args:
            z (Tensor): 7x7x<prototype_dim>
        """

        def l2_convolution(x):
            '''
            Apply prototype vectors as l2-convolution filters on input.

            Translated from (https://github.com/cfchen-duke/ProtoPNet/blob/master/model.py)
            '''
            x2 = x**2
            x2_patch_sum = tf.nn.conv2d(
                input=x2,
                filters=tf.ones([1, 1, x.shape[-1], self.num_prototypes]),
                strides=1,
                padding="SAME")

            p2 = self.prototypes**2
            p2 = tf.math.reduce_sum(p2, axis=-1)
            # p2 is a vector of shape (num_prototypes,)
            # then we reshape it to (num_prototypes, 1, 1)
            p2_reshape = tf.reshape(p2, [self.num_prototypes, 1, 1])

            prototype_filters = tf.expand_dims(
                tf.expand_dims(tf.transpose(self.prototypes), 0), 0)
            xp = tf.nn.conv2d(input=x,
                              filters=prototype_filters,
                              strides=1,
                              padding="SAME")
            intermediate_result = -2 * tf.transpose(p2_reshape + tf.transpose(
                xp, [0, 3, 1, 2]), [0, 2, 3, 1])  # use broadcast
            # x2_patch_sum and intermediate_result are of the same shape
            distances = tf.nn.relu(x2_patch_sum + intermediate_result)
            return distances

        distances = l2_convolution(z)
        similarities = tf.math.log((distances + 1) / (distances + epsilon))

        return distances, similarities
