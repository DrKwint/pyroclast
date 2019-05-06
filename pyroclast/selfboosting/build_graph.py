"""
Implementation of `Learning Deep ResNet Blocks Sequentially using Boosting
Theory`
"""
import tensorflow as tf
import sonnet as snt

from pyroclast.selfboosting.models import ResNet
from pyroclast.selfboosting.modules import get_module

EPS = 1e-10


class HypothesisModule(snt.AbstractModule):
    """Calculates o_t"""

    def __init__(self, class_num, use_bias=False, name='hypothesis_module'):
        """
        Args:
            class_num (int): number of classes in classification problem
            use_bias (bool, optional): controls whether to use a bias term in
                the classifier
        """
        super(HypothesisModule, self).__init__(name=name)
        with self._enter_variable_scope():
            self._classifier = snt.Linear(class_num, use_bias=use_bias)

    def _build(self, inputs):
        """
        Args:
            inputs (tf.Tensor): output of a residual block

        Returns:
            tf.Tensor: classification hypothesis
        """
        return self._classifier(snt.BatchFlatten()(inputs))


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
                    snt.Conv2D(representation_channels, 3),
                    HypothesisModule(class_num),
                    name='initial_boosting_module')
            ]

    def _build(self, x):
        weak_classifiers = []
        hypotheses = [0.]
        alpha = 0.
        hypothesis = 0.
        for module in self.boosting_modules:
            x, hypothesis, weak = module(x, alpha, hypothesis)
            weak_classifiers.append(weak)
            hypotheses.append(hypothesis)
            alpha = module.alpha
        final_logits = (1. / (alpha + EPS)) * tf.add_n(weak_classifiers)
        return final_logits, hypotheses, weak_classifiers

    def add_module(self, module_name):
        base_module = get_module(module_name)(channels=self._channels)
        residual_module = snt.Residual(base_module)
        module = ResidualBoostingModule(residual_module,
                                        HypothesisModule(self._class_num))
        self.boosting_modules.append(module)
        return module

    def get_hypothesis_loss(self, alpha, hypothesis, label):
        """
        Args:
            alpha:
            hypothesis (Tensor):
            label (Tensor): dtype int, shape (batch_size,)
        """
        # Extract vector of logits in the correct class over the batch
        batch_size = tf.cast(tf.shape(label)[0], tf.int64)
        label_idxs = tf.stack([tf.range(batch_size), label], axis=-1)
        label_prediction_vals = tf.gather_nd(hypothesis, label_idxs)
        # Use that to calculate outer loss term (from Eqn. 65)
        outer_loss_term = tf.exp(-alpha * label_prediction_vals)

        # for the inner, sum everything and subtract the term for correct labels
        inner_loss_term = tf.reduce_sum(
            tf.exp(alpha * hypothesis), axis=1) - tf.exp(
                alpha * label_prediction_vals)
        batch_loss = outer_loss_term * inner_loss_term
        loss = tf.reduce_sum(batch_loss, axis=0)
        return loss

    def get_module_train_op(self, optimizer, module, loss):
        vars = module.get_all_variables()
        train_op = optimizer.minimize(loss, var_list=vars)
        return train_op
