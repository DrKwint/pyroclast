import functools

import numpy as np
import tensorflow as tf

from pyroclast.prototype.prototype_layer import PrototypeLayer


class ProtoPNet(tf.Module):
    """Implementation of the classifier introduced in 'This Looks Like That:
    Deep Learning for Interpretable Image Recognition'"""

    def __init__(self,
                 conv_stack,
                 num_prototypes,
                 prototype_dim,
                 num_classes,
                 class_specific=0):
        """
        Args:
            conv_stack (tf.Module): Convolutional network which ends in a 7x7xC representation where C is arbitrary, but typically equal to the prototype_dim
            num_prototypes (int): Number of prototype vectors in the model
            prototype_dim (int): Dimensionality of each prototype vector
            num_classes (int): Number of classes in the classification problem
            class_specific (bool): Optional, whether to use class-specific prototypes and objective
        """
        self.prototype_dim = prototype_dim
        self.class_specific = class_specific

        # Uniform [0,1] init for prototypes as in paper
        self.prototypes = tf.Variable(tf.random.uniform(
            [num_prototypes, prototype_dim]),
                                      trainable=True,
                                      name='prototype_vectors')

        # set the modules needed for forward pass
        self.conv_stack = conv_stack
        self.final_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(prototype_dim,
                                   1,
                                   1,
                                   'same',
                                   activation=tf.nn.relu),
            tf.keras.layers.Conv2D(prototype_dim,
                                   1,
                                   1,
                                   'same',
                                   activation=tf.nn.sigmoid)
        ])
        self.prototype_layer = PrototypeLayer(num_prototypes, prototype_dim)
        self.max_pool = tf.keras.layers.GlobalMaxPool2D()

        if class_specific:
            # set prototype class identity
            num_prototypes_per_class = num_prototypes // num_classes
            self.prototype_class_identity = np.zeros(
                [num_prototypes, num_classes], dtype=np.float32)
            for j in range(num_prototypes):
                self.prototype_class_identity[j, j //
                                              num_prototypes_per_class] = 1

            # create classifier
            classifier_kernel_init = lambda shape, dtype: -0.5 * tf.ones(
                shape, dtype=dtype) + 1.5 * self.prototype_class_identity
            self.classifier = tf.keras.layers.Dense(
                num_classes,
                use_bias=False,
                kernel_initializer=classifier_kernel_init)
        else:
            self.classifier = tf.keras.layers.Dense(num_classes, use_bias=False)

    @property
    def trainable_prototype_vars(self):
        return list(self.prototype_layer.trainable_variables)

    @property
    def final_conv_vars(self):
        self.final_conv.trainable_variables

    @property
    def trainable_conv_stack_vars(self):
        return self.conv_stack.trainable_variables

    @property
    def trainable_classifier_vars(self):
        return self.classifier.trainable_variables

    def __call__(self, x):
        """
        Args:
            x (Tensor): image data to be classified

        Returns:
            Tuple of tensors (y_hat, minimum distances, conv_output)
        """
        conv_output = self.final_conv(self.conv_stack(x))
        assert conv_output.shape[1:3] == [7, 7]
        distances, similarities = self.prototype_layer(conv_output,
                                                       self.prototypes)
        minimum_distances = -self.max_pool(-distances)
        prototype_activations = self.max_pool(similarities)
        return self.classifier(
            prototype_activations), minimum_distances, conv_output

    def conv_prototype_objective(self, min_distances, label=None):
        """
        Args:
            min_distances (tf.Tensor): shape [batch_size, num_prototypes] values calculated by self.__call__
            label (tf.Tensor): shape [batch_size] correct labels from dataset

        Returns:
            dict of string to tf.Tensor, objective components of shape [batch_size]
            May contain keys ['cluster', 'separation', and 'l1']
        """
        term_dict = dict()
        if self.class_specific:
            assert label is not None
            prototypes_of_correct_class = tf.transpose(
                self.prototype_class_identity[:, label])
            """
            Because the last activation of `self.final_conv` is a sigmoid,
            this is the numerical maximum. The only problem with this
            interpretation is that the norm of each prototype vector is
            unbounded and the l2 distance between it and an image patch could
            be greater than 1. Prototypes are initialized in the range [0,1].
            """
            max_dist = self.prototype_dim
            inverted_distances = tf.reduce_max(
                (max_dist - min_distances) * prototypes_of_correct_class,
                axis=1)
            term_dict['cluster'] = max_dist - inverted_distances

            # calculate separation cost
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            inverted_distances_to_nontarget_prototypes = (
                max_dist - min_distances) * prototypes_of_wrong_class
            term_dict['separation'] = tf.reduce_mean(
                max_dist - inverted_distances_to_nontarget_prototypes, axis=1)
        else:
            min_distance = tf.math.reduce_min(min_distances, axis=1)
            term_dict['cluster'] = tf.math.reduce_mean(min_distance)

        term_dict['l1'] = tf.norm(self.classifier.trainable_weights[0], 1)
        return term_dict
