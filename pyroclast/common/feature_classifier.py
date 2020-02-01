import abc
import numpy as np
import tensorflow as tf

from pyroclast.common.adversarial import fast_gradient_sign_method
import sonnet as snt


class FeatureClassifierMixin(abc.ABC):
    """Feature Classifier Mixin.

    A mixin that provides base implementations for models that work by
    first calculating features via a conv stack and then performing
    classification based off those features. Formalizing this much of
    the network allows us to perform `usefulness' and `robustness'
    analysis on the features.

    Definitions of 'usefulness' and 'robustness' from :ref:`Adversarial Examples Are
    Not Bugs, They Are Features by Ilyas et al.<https://arxiv.org/abs/1905.02175>`
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

    def usefulness(self, D):
        """Finds maximal rho such that f is rho-useful over the dataset D for
        classifying y

        Args:
           D (tf.data.Dataset): A dataset where `image' and `label' are valid keys for each datam. `image' should be f32 and shape [N..HWC], `label' can be either an int or one-hot encoding.

        Returns:
           rho (tf.Tensor): The usefulness of each feature for each class. Of shape [num_features, num_classes].
        """
        def get_one_hot(x):
            return tf.cast(
                tf.one_hot(x,
                           num_classes,
                           on_value=1,
                           off_value=-1), tf.float32)

        def cast_and_get_features(x):
            x = tf.cast(x, tf.float32)
            features = self.features
            if len(x.shape) > 4:
                merge_dims = len(x.shape) - 3
                features = snt.BatchApply(self.features, merge_dims)
            return features(x)

        for d in D:
            features = cast_and_get_features(d['image'])
            num_classes = self.classify_features(features).shape[-1]
            break

        def calc_fcdot(x, y):
            """
            Args:
                x (Tensor): f32 data with shape [N..HWC]
                y (Tensor): int labels
            """
            features = cast_and_get_features(x)
            binary_labels = get_one_hot(y)
            einsum = tf.linalg.matmul(tf.expand_dims(features, -1), tf.expand_dims(binary_labels, -2))
            return einsum


        rho = D.map(lambda x: calc_fcdot(x['image'], x['label']))
        rho, num_data = rho.reduce((0., 0), lambda x, y: (x[0] + tf.reduce_sum(y, axis=0), x[1] + tf.shape(y)[0]))
        rho = rho / float(num_data)
        return rho

    def robustness(self, D, eps, norm):
        """Calculates the robustness of features in a network with respect to
        a dataset D and a perturbation class defined by norm and eps.

        Args:
           D (tf.data.Dataset): A dataset where `image' and `label' are valid keys for each datum
           eps (float): numerical limit on the norm of the actual delta
           norm (1, 2, or np.inf): Which class of delta to use

        Returns:
           gamma (tf.Tensor): The robustness of each feature for each class. Of shape [num_features, num_classes].
        """
        # get number of classes
        for d in D:
            features = self.features(tf.cast(d['image'], tf.float32))
            num_classes = self.classify_features(features).shape[-1]
            break

        def get_one_hot(x):
            return tf.cast(tf.one_hot(x, num_classes, on_value=1, off_value=-1),
                           tf.float32)

        # create adversarially perturbed dataset and calulate its usefulness
        D_adv = D.map(lambda x: {'image': fast_gradient_sign_method(
            self.features, self.classify_features,
            tf.cast(x['image'], tf.float32), get_one_hot(x['label']), 0.01, 1),
                                'label': tf.expand_dims(tf.expand_dims(x['label'], 1), 1)})
        adv_usefulness = self.usefulness(D_adv)

        # create a mask with 1's where the class and feature line up in both
        # the data portion (first 2 dims) and the calulated usefulness (last 2 dims)
        # because we were only adversarially attacking one class/feature pair at a time
        idxs = np.array([[[i,j,i,j] for j in range(adv_usefulness.shape[1])] for i in range(adv_usefulness.shape[0])])
        idxs = np.reshape(idxs, [-1, 4])
        mask = np.zeros_like(adv_usefulness)
        mask[tuple(idxs.T)] = 1.
        # apply mask and reduce over the calulated usefulness dims (last 2)
        adv_usefulness = tf.reduce_sum(tf.reduce_sum(adv_usefulness * mask, -1), -1)
        return adv_usefulness
