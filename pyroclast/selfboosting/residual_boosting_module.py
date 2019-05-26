"""
Implementation of `Learning Deep ResNet Blocks Sequentially using Boosting
Theory`
"""
import sonnet as snt
import tensorflow as tf

from pyroclast.selfboosting.modules import get_module


class ResidualBoostingModule(snt.AbstractModule):
    def __init__(self,
                 repr_module_name,
                 classification_module_name,
                 num_channels,
                 num_classes,
                 name='residual_boosting_module'):
        """
        Args:
            repr_module_name (str): name of a non-residual block to calculate
                next residual representation
            hypothesis_module_name (str): name of a classification block
        """
        super(ResidualBoostingModule, self).__init__(name=name)
        with self._enter_variable_scope():
            repr_module = get_module(repr_module_name)(channels=num_channels)
            self.residual_module = snt.Residual(repr_module)
            self.hypothesis_module = get_module(classification_module_name)(
                classes=num_classes)
            self.alpha = tf.get_variable("alpha", shape=())

    def _build(self, data, prev_alpha, prev_hypothesis):
        """
        Args:
            data (Tensor): rank 4 NHWC input representation
            prev_alpha (Tensor): alpha from previous module
            prev_hypothesis (Tensor): hypothesis from previous module

        Returns:
            residual_repr (Tensor): rank 4 representation in ResNet
            hypothesis (Tensor): weak learner classification
            boosted_classification (Tensor): rank 4 output of boosted learner
                to this point
        """
        residual_repr = self.residual_module(data)
        hypothesis = self.hypothesis_module(snt.BatchFlatten()(residual_repr))
        boosted_classification = (self.alpha * hypothesis) - (
            prev_alpha - prev_hypothesis)
        return residual_repr, hypothesis, boosted_classification

    def initialize_variables(self, session, optimizer):
        session.run(tf.initialize_variables(self.get_all_variables()))
        session.run(tf.initialize_variables(optimizer.variables()))

    def get_train_op(self, optimizer, loss):
        """ Gets a tf.Operation which updates only this module's variables.

        Args:
            optimizer (Optimizer): any optimizer
            loss (Tensor): scalar value to minimize
        """
        vars = self.get_all_variables()
        train_op = optimizer.minimize(loss, var_list=vars)
        return train_op
