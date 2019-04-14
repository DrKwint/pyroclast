import tensorflow as tf
import sonnet as snt

import pyroclast.classification.selfboosting.models as models


class HypothesisModule(object):
    """Calculates o_t"""

    def __init__(self, class_num, use_bias=False):
        """
        Args:
            class_num (int): number of classes in classification problem
            use_bias (bool, optional): controls whether to use a bias term in
                the classifier
        """
        self._classifier = snt.Linear(class_num, use_bias=use_bias)

    def _build(self, inputs):
        """
        Args:
            inputs (tf.Tensor): output of a residual block

        Returns:
            tf.Tensor: classification hypothesis
        """
        return self._classifier(inputs)


class WeakModuleClassifier(object):
    """Calculates h_t

    Attributes:
        alpha_tp1 (tf.Tensor): alpha owned by this module
    """

    def __init__(self, alpha_t=None, alpha_shape=()):
        """
        Args:
            alpha_t (tf.Tensor): alpha value owned by previous
                WeakModuleClassifier. Should only be None if this is the first
                weak module.
            alpha_shape (int tuple): shape of alpha_tp1 (we expect 0 or 1 dim)
        """
        # guarantee that alpha_shape is sensible
        assert len(alpha_shape) <= 1
        if alpha_t is not None:
            assert alpha_shape.shape().as_list() == list(alpha_shape)
        # set alpha_t
        if alpha_t:
            self._alpha_t = alpha_t
        else:
            self._alpha_t = tf.get_variable(
                "alpha_t", shape=alpha_shape, initializer=tf.zeros_initializer)
        # initialize alpha_{t+1}
        self.alpha_tp1 = tf.get_variable("alpha_t+1", shape=alpha_shape)

    def _build(self, inputs):
        hypothesis_t, hypothesis_tp1 = inputs
        return (self._alpha_tp1 * hypothesis_tp1) - (
            self._alpha_t * hypothesis_t)


class WeakChainClassifier(snt.Module):
    """Calculates full ResNet output with telescoping boosting"""

    def __init__(self, chain_length, alpha_shape=()):
        """hypothesis_modules: [HypothesisModule]"""
        self._module_chain = [WeakModuleClassifier(alpha_shape=alpha_shape)]
        for _ in range(1, chain_length):
            alpha_t = self._module_chain[-1].get_alpha_tp1()
            self._module_chain.append(
                WeakModuleClassifier(alpha_t, alpha_shape))

    def _build(self, inputs):
        """
        Args:
            inputs: list of hypothesis modules (o_t in the paper)
        """
        weak_module_outputs = self.calculate_weak_modules(inputs)
        final_classification = (
            1. / self._module_chain[-1].get_alpha_tp1()) * tf.reduce_sum(
                weak_module_outputs, axis=1)
        return final_classification

    def calculate_weak_modules(self, inputs):
        assert len(inputs) == len(self._module_chain)
        hypotheses = [tf.zeros_like(inputs[0])] + inputs
        weak_modules = [
            module(hypotheses[t], hypotheses[t + 1])
            for t, module in enumerate(self._module_chain)
        ]
        return weak_modules


def build_model(data_tensor, repr_channels, module_name, module_num):
    initial_trans = snt.Conv2D(repr_channels, 3)
    modules = [
        models.modules[module_name](repr_channels) for _ in range(module_num)
    ]
    network = models.ResNet(initial_trans, modules)
