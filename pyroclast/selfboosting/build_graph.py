import tensorflow as tf
import sonnet as snt

from pyroclast.selfboosting.models import ResNet
from pyroclast.selfboosting.modules import get_module


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


class WeakModuleClassifier(snt.AbstractModule):
    """Calculates h_t

    Attributes:
        alpha_tp1 (tf.Tensor): alpha owned by this module
    """

    def __init__(self,
                 alpha_t=None,
                 alpha_shape=(),
                 name='weak_module_classifier'):
        """
        Args:
            alpha_t (tf.Tensor): alpha value owned by previous
                WeakModuleClassifier. Should only be None if this is the first
                weak module.
            alpha_shape (int tuple): shape of alpha_tp1 (we expect 0 or 1 dim)
        """
        super(WeakModuleClassifier, self).__init__(name=name)
        # guarantee that alpha_shape is sensible
        assert len(alpha_shape) <= 1
        if alpha_t is not None:
            assert alpha_t.shape.as_list() == list(alpha_shape)
        with self._enter_variable_scope():
            # set alpha_t
            if alpha_t:
                self._alpha_t = alpha_t
            else:
                self._alpha_t = tf.get_variable(
                    "alpha_t",
                    shape=alpha_shape,
                    initializer=tf.zeros_initializer)
            # initialize alpha_{t+1}
            self.alpha_tp1 = tf.get_variable("alpha_tp1", shape=alpha_shape)

    def _build(self, inputs):
        hypothesis_t, hypothesis_tp1 = inputs
        return (self.alpha_tp1 * hypothesis_tp1) - (
            self._alpha_t * hypothesis_t)


class WeakChainClassifier(snt.AbstractModule):
    """Calculates full ResNet output with telescoping boosting"""

    def __init__(self,
                 chain_length,
                 alpha_shape=(),
                 name="weak_chain_classifier"):
        """hypothesis_modules: [HypothesisModule]"""
        super(WeakChainClassifier, self).__init__(name=name)
        self._module_chain = [WeakModuleClassifier(alpha_shape=alpha_shape)]
        for _ in range(1, chain_length):
            alpha_t = self._module_chain[-1].alpha_tp1
            self._module_chain.append(
                WeakModuleClassifier(alpha_t, alpha_shape))

    def _build(self, inputs):
        """
        Args:
            inputs: list of hypothesis modules (o_t in the paper)
        """
        weak_module_outputs = self.calculate_weak_modules(inputs)
        final_classification = (
            1. / self._module_chain[-1].alpha_tp1) * tf.reduce_sum(
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


class ResidualBoostingModule(snt.AbstractModule):
    def __init__(self,
                 residual_module,
                 hypothesis_module,
                 name='residual_boosting_module'):
        super(ResidualBoostingModule, self).__init__(name=name)
        self.residual_module = residual_module
        self.hypothesis_module = hypothesis_module
        with self._enter_variable_scope():
            self.alpha = tf.get_variable("alpha", shape=())

    def _build(self, data, prev_alpha, prev_hypothesis):
        output = self.residual_module(data)
        hypothesis = self.hypothesis_module(output)
        weak_classification = (self.alpha * hypothesis) - (
            prev_alpha - prev_hypothesis)
        return output, hypothesis, weak_classification


class SequentialResNet(snt.AbstractModule):
    def __init__(self,
                 class_num,
                 representation_channels,
                 name='sequential_resnet'):
        super(SequentialResNet, self).__init__(name=name)
        self._class_num = class_num
        self._channels = representation_channels
        with self._enter_variable_scope():
            self.base_hypothesis = tf.get_variable(
                'base_hypothesis', shape=(class_num))
            self.boosting_modules = [
                ResidualBoostingModule(
                    snt.Conv2D(32, 3), HypothesisModule(class_num))
            ]

    def _build(self, x):
        data, hypothesis, first_weak = self.boosting_modules[0]._build(
            x, 1., self.base_hypothesis)
        alpha = self.boosting_modules[0].alpha
        weak_classifiers = [first_weak]
        for module in enumerate(self.boosting_modules):
            data, hypothesis, weak = module(data, alpha, hypothesis)
            weak_classifiers.append(weak)
            alpha = module.alpha
        final_classification = (1. / alpha) * tf.add_n(weak_classifiers)
        return final_classification, weak_classifiers

    def add_module(self, module_name):
        base_module = get_module(module_name)(channels=self._channels)
        residual_module = snt.Residual(base_module)
        module = ResidualBoostingModule(residual_module,
                                        HypothesisModule(self._class_num))
        self.boosting_modules.append(module)
        return module

    def get_module_loss(self, optimizer, weak_learner, label):
        vars = weak_learner.get_all_variables()

        loss = tf.exp()
        optimizer.minimize(loss, var_list=vars)
