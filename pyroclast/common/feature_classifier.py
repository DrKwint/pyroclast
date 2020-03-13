import abc

import numpy as np
import sonnet as snt
import tensorflow as tf

from pyroclast.common.adversarial import fast_gradient_method
from pyroclast.common.tf_util import OnePassCorrelation


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

    @abc.abstractmethod
    def get_classification_module(self):
        """Returns the variables in the final classification layer

        returns:
           feature_variables (tf.Module): The classification module
        """
        pass

    def feature_importance_for_class(self, data_dict, class_idx):
        """Finds the relative importance of features to a given class

        Note: Sorting is in the descending direction.

        returns:
           feature_ids (tf.Tensor): The ids of the features, sorted by weight in dense layer.
           weights (tf.Tensor): The weights of each feature, unsorted.
        """
        classifier_module = self.get_classification_module()
        dense_kernel = list(
            filter(lambda v: 'dense' in v.name and 'kernel' in v.name,
                   classifier_module.variables))[-1]
        weights = tf.squeeze(dense_kernel[:, class_idx])
        feature_ids = tf.argsort(weights, direction='DESCENDING')
        return (feature_ids, weights)

    def usefulness(self, iterable, num_classes, is_preprocessed=False):
        """Finds maximal rho per feature/class pair such that the model is rho-useful over the dataset D

        Args:
           D (tf.data.Dataset): A dataset where `image' and `label' are valid keys for each datam. `image' should be f32 and shape [N..HWC], `label' can be either an int or one-hot encoding.

        Returns:
           rho (tf.Tensor): The usefulness of each feature for each class. Of shape [num_features, num_classes].
        """

        def get_features(x):
            if is_preprocessed:
                return x
            features = self.features
            if len(x.shape) > 4:
                merge_dims = len(x.shape) - 3
                features = snt.BatchApply(self.features, merge_dims)
            elif len(x.shape) < 4:
                x = tf.expand_dims(x, 0)
            return features(x)

        def get_one_hot(x, num_classes):
            return tf.cast(tf.one_hot(x, num_classes, on_value=1, off_value=-1),
                           tf.float32)

        corr_calc = OnePassCorrelation()
        for x, y in iterable:
            labels = tf.expand_dims(get_one_hot(y, num_classes), -2)
            features = tf.expand_dims(get_features(x), -1)
            corr_calc.accumulate(labels, features)

        return corr_calc.finalize()

    def robustness(self, iterable, feature_idx, class_idx, num_classes, eps,
                   norm):
        """Calculates the robustness of features in a network with respect to
        a dataset D and a perturbation class defined by norm and eps.

        Args:
           D (tf.data.Dataset): A dataset where `image' and `label' are valid keys for each datum
           eps (float): numerical limit on the norm of the actual delta
           norm (1, 2, or np.inf): Which class of delta to use

        Returns:
           gamma (tf.Tensor): The robustness of each feature for each class. Of shape [num_features, num_classes].
        """

        def get_one_hot(x, num_classes):
            return tf.cast(tf.one_hot(x, num_classes, on_value=1, off_value=-1),
                           tf.float32)

        # create adversarially perturbed dataset and calulate its usefulness
        def adv_generator(D):
            for x, y in D:
                img = tf.cast(x, tf.float32)
                labels = get_one_hot(y, num_classes)
                forward_fn = lambda x: self.features(x)[:, feature_idx
                                                       ] * labels[:, class_idx]
                adv_img = img + fast_gradient_method(forward_fn, img, eps, norm)
                yield adv_img, y

        adv_usefulness = self.usefulness(adv_generator(iterable), num_classes)
        return adv_usefulness

    def input_search(self,
                     x,
                     feature_target,
                     early_stopping,
                     search_method='fgm'):
        if search_method == 'fgm':
            forward_fn = lambda _x: tf.norm(feature_target - self.features(_x),
                                            2)  # + 0.1 * tf.norm(x - _x, 2)
            for i in range(early_stopping.max_epochs):
                delta = fast_gradient_method(forward_fn, x, 0.0001, 2)
                x += delta
                if early_stopping(i, forward_fn(x)):
                    break
        else:
            raise NotImplementedError()
        return x
