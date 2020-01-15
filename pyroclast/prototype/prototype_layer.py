import tensorflow as tf


class PrototypeLayer(tf.Module):

    def __init__(self, num_prototypes, prototype_dim):
        """
        Args:
            num_prototypes (int): Number of prototype vectors in the model
            prototype_dim (int): Dimensionality of each prototype vector
        """
        self.prototype_dim = prototype_dim
        self.num_prototypes = num_prototypes
        # Uniform [0,1] init for prototypes as in paper
        self.prototypes = tf.Variable(tf.random.uniform(
            [num_prototypes, prototype_dim]),
                                      trainable=True)
        self.max_pool = tf.keras.layers.GlobalMaxPool2D()

    def __call__(self, z, epsilon=1e-4):
        """
        Args:
            z (Tensor): 7x7x<prototype_dim>
            epsilon (float): Optional, some sufficiently small value
        """

        def l2_convolution(images, vectors):
            '''
            Calculate pairwise l2 distance between the patches of an image and an array of vectors

            Calculating the l2 distance between each of the HxW patches and the
            N_2 vectors. Naively, this would require looping O(n^2) times, but
            a convolution makes the control flow moot.

            Args:
                images (tf.Tensor): shape [N_1,H,W,C] activation images
                vectors (tf.Tensor): shape [N_2,C] list of vectors

            Returns:
                tf.Tensor of l2 distances with shape [N_1, H, W, N_2]
            '''
            # shape [N_1,H,W,N_2]
            image_sq = tf.nn.conv2d(
                input=images**2,
                filters=tf.ones([1, 1, images.shape[-1], vectors.shape[0]]),
                strides=1,
                padding="SAME")

            # shape [N_2]
            vectors_sq = tf.reduce_sum(vectors**2, 1)

            vector_filters = tf.expand_dims(
                tf.expand_dims(tf.transpose(vectors), 0), 0)
            # shape [N_1,H,W,N_2]
            image_vector_prod = tf.nn.conv2d(input=images,
                                             filters=vector_filters,
                                             strides=1,
                                             padding="SAME")
            # equiv to $(x - y)^2$
            return image_sq - (2 * image_vector_prod) + vectors_sq

        distances = l2_convolution(z, self.prototypes)
        similarities = tf.math.log((distances + 1) / (distances + epsilon))

        return distances, similarities
