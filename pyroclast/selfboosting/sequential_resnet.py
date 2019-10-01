import sonnet as snt
import tensorflow as tf

from pyroclast.selfboosting.residual_boosting_module import ResidualBoostingModule

EPS = 1e-10


class SequentialResNet(snt.AbstractModule):
    """
    Implementation of boosted residual network introduced in Huang et al. 2017

    Usage steps:
        1) SequentialResNet() to create module
        2) `build()` to initialize network with input data
        3) `add_module()` to add a ResidualBoostingModule to network
        4) `get_hypothesis_loss()` to get loss of the new module's hypothesis
        5) `module.get_module_train_op()` to get the train_op which optimizes
            only that module
        6) Train to satisfaction and then GOTO step 3 until done

    Public mutable state:
        boosting_modules: list of boosting modules belonging to the network,
            updated by `add_module()`

    Private immutable state:
        _base_hypothesis (Variable): (currently trainable)
        _base_alpha (Tensor): associated with base hypothesis (currently zero)
        _base_repr_module (snt.Module): conv which gets data to num of channels

    Private mutable state:
        _last_repr (Tensor): last residual representation
        _last_alpha (Tensor): last alpha
        _last_hypothesis (Tensor):
    """

    def __init__(self,
                 class_num,
                 representation_channels,
                 name='sequential_resnet'):
        """
        Args:
            class_num (int): number of classes in classification problem
            representation_channels (int): number of channels for residual
                representation
        """
        super(SequentialResNet, self).__init__(name=name)
        self.boosting_modules = []
        with self._enter_variable_scope():
            # trainable base hypothesis, maybe update if we get spooky action
            self._base_hypothesis = tf.get_variable('base_hypothesis',
                                                    shape=(class_num))
            self._base_alpha = tf.zeros(class_num)
            self._base_repr_module = snt.Conv2D(representation_channels,
                                                3,
                                                name='base_repr_conv2d')
            # private state variables cached for adding modules
            self._last_repr = tf.zeros(())
            self._last_alpha = self._base_alpha
            self._last_hypothesis = self._base_hypothesis
            # last module can be calculated from self.boosting_modules

    def _build(self, x, is_train=False):
        """
        Builds ResNet with all modules and returns final and intermediate
        classifications. Should be called first thing after `__init__` or to
        retrieve intermediate classifications.

        Updates private state variables if `is_train` is True.

        Returns:
            final_logits (Tensor): final boosted classification
            hypotheses (list): output of weak learners
            boosted_classifications (list): intermediate boosted classifications
        """
        weak_classifiers = []
        hypotheses = [0.]
        alpha = 0.
        hypothesis = 0.

        # build network
        x = self._base_repr_module(x)
        for module in self.boosting_modules:
            x, hypothesis, weak = module(x, alpha, hypothesis)
            weak_classifiers.append(weak)
            hypotheses.append(hypothesis)
            alpha = module.alpha

        # update final value cache
        if is_train:
            self._last_repr = x

        # handle case where add_module has not been called yet
        if weak_classifiers:
            final_logits = (1. / (alpha + EPS)) * tf.add_n(weak_classifiers)
        else:
            final_logits = self._base_hypothesis

        return final_logits, hypotheses, weak_classifiers

    def add_module(self, residual_boosting_module):
        """
        Adds a ResidualBoostingModule to the network and initializes its
        variables. Must only be called after `build`.

        Updates list of modules and private cache of last calculated values.

        Args:
            residual_boosting_module (ResidualBoostingModule)
            optimizer: optimizer whose variables relate only to the new module
            session (tf.Session)

        Returns:
            alpha (Tensor): new module's alpha variable
            hypothesis (Tensor): new module's weak classification hypothesis
            boosted_classification (Tensor): network's final classification

        Raises:
            snt.NotConnectedError: If `build` hasn't been called before this
        """
        self._ensure_is_connected()

        # add a ResidualBoostingModule to module list
        self.boosting_modules.append(residual_boosting_module)

        # build module with cached last values to get outputs
        residual_repr, hypothesis, boosted_classification = residual_boosting_module(
            self._last_repr, self._last_alpha, self._last_hypothesis)
        # residual_boosting_module.initialize_variables(session, optimizer)

        # update cache of calculated values
        self._last_repr = residual_repr
        self._last_alpha = residual_boosting_module.alpha
        self._last_hypothesis = hypothesis

        return residual_boosting_module.alpha, hypothesis, boosted_classification

    def get_hypothesis_loss(self, alpha, hypothesis, label):
        """Calculates loss (without state)
        Args:
            alpha (Tensor): module alpha associated with hypothesis
            hypothesis (Tensor): weak learner classification w/label shape
            label (Tensor): dtype int, shape (batch_size,)
        """
        # Extract vector of logits in the correct class over the batch
        batch_size = tf.cast(tf.shape(label)[0], tf.int64)
        label_idxs = tf.stack([tf.range(batch_size), label], axis=-1)
        label_prediction_vals = tf.gather_nd(hypothesis, label_idxs)
        # Use that to calculate outer loss term (from Eqn. 65)
        outer_loss_term = tf.exp(-alpha * label_prediction_vals)

        # for the inner, sum everything and subtract the term for correct labels
        inner_loss_term = tf.reduce_sum(tf.exp(
            alpha * hypothesis), axis=1) - tf.exp(alpha * label_prediction_vals)
        batch_loss = outer_loss_term * inner_loss_term
        loss = tf.reduce_sum(batch_loss, axis=0)
        return loss
