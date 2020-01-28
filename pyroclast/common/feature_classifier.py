import abc
import numpy as np
import tensorflow as tf


class FeatureClassifierMixin(abc.ABC):
    """Feature Classifier Mixin.

    A mixin that provides base implementations for models that work by
    first calculating features via a conv stack and then performing
    classification based off those features. Formalizing this much of
    the network allows us to perform `usefulness' and `robustness'
    analysis on the features.
    """

    @abc.abstractmethod
    def features(self, x):
        """Returns a flat tensor of features

        This method calculates the features given some input data x.

        Args:
           x (tf.Tensor): Input of shape [batch_size, ...data_shape]

        Returns:
           features (tf.Tensor): Features of each data in the input [batch_size, num_features]
        """
        pass

    @abc.abstractmethod
    def classify_features(self, features):
        """Calculates output logits given a feature tensor

        Args:
           features (tf.Tensor): Input of shape [batch_size, num_features]

        Returns:
           logits (tf.Tensor): Output of shape [batch_size, num_classes]
        """
        pass

    def usefulness(self, D, num_classes, num_features):
        """Finds maximal rho such that f is rho-useful over the dataset D for
        classifying y

        Args:
           D (tf.data.Dataset): A dataset where `image' and `label' are valid keys for each datam. `label' can be either an int or one-hot encoding.
           num_classes (int): The number of classes in the problem
           num_features (int): The number of features output from `features()'. Better way to do this?

        Returns:
           rho (tf.Tensor): The usefulness of each feature for each class. Of shape [num_features, num_classes].
        """

        def class_index(x_i):
            if type(x_i['label']) == int:
                return x_i['label']
            else:
                return tf.math.argmax(x_i['label'])

        def get_features_and_cast(x_i):
            x_i = tf.cast(x_i['image'], tf.float32)
            return self.features(x_i)

        print(D)

        rho = D.map(lambda x: (get_features_and_cast(x),
                               tf.cast(
                                   tf.one_hot(class_index(x),
                                              num_classes,
                                              on_value=1,
                                              off_value=-1), tf.float32)))
        print(rho)
        rho = rho.map(lambda f, c: tf.tensordot(f, c, 0))
        print(rho)
        rho = rho.reduce(tf.zeros([num_features, num_classes]),
                         lambda x, y: x + tf.reduce_sum(y, 0))

        print(rho)

        assert rho.shape == [num_features, num_classes]
        return rho

    def robustness(self, D, num_classes, Delta):
        """Calculates the robustness of features in a network with respect to
        a dataset D and perturbation class Delta.

        Args:
           D (tf.data.Dataset): An unbatched dataset where image and label are valid keys for each datam
           num_classes (int): The number of classes in the problem
           Delta (): A perturbation class

        Returns:
           gamma (tf.Tensor): The robustness of each feature for each class. Of shape [num_features, num_classes].
        """
        pass
