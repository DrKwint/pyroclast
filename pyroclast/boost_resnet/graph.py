import tensorflow as tf

EPS = 1e-10


class ResidualBoostingModule(tf.Module):
    def __init__(self,
                 residual_repr_module,
                 classification_module,
                 alpha=None,
                 name='residual_boosting_module'):
        """
        Args:
            repr_module_name (str): name of a non-residual block to calculate
                next residual representation
            hypothesis_module_name (str): name of a classification block
        """
        super(ResidualBoostingModule, self).__init__(name=name)
        self.residual_module = residual_repr_module
        self.hypothesis_module = classification_module
        if alpha is None:
            self.alpha = tf.Variable(
                tf.random.normal(shape=()),
                name="alpha_{}".format(name),
                shape=())
        else:
            self.alpha = tf.Variable(alpha, name="alpha_{}".format(name))

    def __call__(self, data, prev_alpha, prev_hypothesis):
        """
        Args:
            data (Tensor): rank 4 NHWC input representation
            prev_alpha (Tensor): alpha from previous module
            prev_hypothesis (Tensor): hypothesis from previous module

        Returns:
            residual_repr (Tensor): rank 4 representation in ResNet
            hypothesis (Tensor): weak learner classification
            weak_module_classifier (Tensor): rank 4 alpha * o - (alpha * o)_prev
        """
        residual_repr = self.residual_module(data)
        hypothesis = self.hypothesis_module(
            tf.keras.layers.Flatten()(residual_repr))
        weak_module_classifier = (self.alpha * hypothesis) - (
            prev_alpha - prev_hypothesis)
        return residual_repr, hypothesis, weak_module_classifier


class SequentialResNet(tf.Module):
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
                 initial_distribution_fn,
                 distribution_update_fn,
                 gamma_tilde_calculation_fn,
                 name='sequential_resnet'):
        """
        Args:
            class_num (int): number of classes in classification problem
            representation_channels (int): number of channels for residual
                representation
        """
        super(SequentialResNet, self).__init__(name=name)
        self.class_num = class_num
        self.boosting_modules = []
        self._base_repr_module = tf.keras.layers.Conv2D(
            representation_channels,
            3,
            padding='same',
            name='base_repr_conv2d')
        self.initial_distribution_fn = initial_distribution_fn
        self.distribution_update_fn = distribution_update_fn
        self.gamma_tilde_calculation_fn = gamma_tilde_calculation_fn

    def __call__(self, x, y=None):
        """
        Returns:
            final_logits (Tensor): final boosted classification
            hypotheses (list): output of each weak learners
            boosted_classifications (list): intermediate boosted classifications
        """
        hypothesis = 0.
        hypotheses = [tf.zeros(self.class_num)]
        weak_module_classifier = 0.
        weak_module_classifiers = []
        if y is not None:
            distribution = self.initial_distribution_fn(y)
            distributions = [distribution]
            gamma_tildes = [1.]
            gammas = []
        alpha = tf.zeros(self.class_num)
        alphas = [alpha]
        boosted_classifiers = [tf.zeros([1, self.class_num])]

        # build network
        x = self._base_repr_module(x)
        for module in self.boosting_modules:
            x, hypothesis, weak_module_classifier = module(
                x, alpha, hypothesis)
            alpha = module.alpha

            if y is not None:
                gamma_tildes.append(
                    self.gamma_tilde_calculation_fn(
                        y, hypothesis, hypotheses[-1], distributions[-1],
                        boosted_classifiers[-1]))
                gammas.append(
                    tf.sqrt((tf.square(gamma_tildes[-1]) - tf.square(
                        gamma_tildes[-2])) / 1. - tf.square(gamma_tildes[-2])))
                distribution = self.distribution_update_fn(
                    weak_module_classifier, y, distribution)
                distributions.append(distribution)

            hypotheses.append(hypothesis)
            weak_module_classifiers.append(weak_module_classifier)
            alphas.append(alpha)
            try:
                boosted_classifiers.append(
                    boosted_classifiers[-1] + weak_module_classifier)
            except IndexError:
                boosted_classifiers.append(0.)

        return boosted_classifiers, weak_module_classifiers, gamma_tildes, gammas, alphas

    def add_module(self, residual_boosting_module):
        # add a ResidualBoostingModule to module list
        self.boosting_modules.append(residual_boosting_module)
