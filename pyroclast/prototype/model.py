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
            conv_stack (Module): Convolutional network which ends in a 7x7xC representation where C is arbitrary
        """
        self.class_specific = class_specific

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
            num_prototypes_per_class = self.num_prototypes // self.num_classes
            self.prototype_class_identity = tf.zeros(self.num_prototypes,
                                                     self.num_classes)
            for j in range(self.num_prototypes):
                self.prototype_class_identity[j, j //
                                              num_prototypes_per_class] = 1

            # create classifier
            classifier_kernel_init = lambda shape: -0.5 * tf.ones(
                shape) + 1.5 * self.prototype_class_identity
            self.classifier = tf.keras.layers.Dense(
                num_classes,
                use_bias=False,
                kernel_initializer=classifier_kernel_init)
        else:
            self.classifier = tf.keras.layers.Dense(num_classes, use_bias=False)

    def __call__(self, x):
        """
        Args:
            x (Tensor): image data to be classified
        """
        conv_output = self.final_conv(self.conv_stack(x))
        distances, similarities = self.prototype_layer(conv_output)
        minimum_distances = -self.max_pool(-distances)
        prototype_activations = self.max_pool(similarities)
        return self.classifier(prototype_activations), minimum_distances

    def conv_prototype_objective(self, min_distances):
        term_dict = dict()
        if self.class_specific:
            raise NotImplementedError()
        else:
            min_distance = tf.math.reduce_min(min_distances, axis=1)
            term_dict['cluster'] = tf.math.reduce_mean(min_distance)
            term_dict['l1'] = tf.norm(self.classifier.trainable_weights[0], 1)
        return term_dict
