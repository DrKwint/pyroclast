import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from pyroclast.cpvae.ddt import DDT

tfd = tfp.distributions


def build_linear_classifier(num_classes, class_loss_coeff):
    return DistributionClassifier(
        snt.Sequential([snt.Flatten(), snt.Linear(num_classes)]),
        class_loss_coeff)


def build_nonlinear_classifier(num_classes, class_loss_coeff):
    return DistributionClassifier(
        snt.Sequential([
            snt.Flatten(),
            snt.Linear(num_classes * 3), tf.nn.leaky_relu,
            snt.Linear(num_classes)
        ]), class_loss_coeff)


def build_ddt_classifier(max_depth, class_loss_coeff):
    return DistributionClassifier(DDT(max_depth, use_analytic=False),
                                  class_loss_coeff)


class DistributionClassifier(tf.Module):

    def __init__(self, classifier, class_loss_coeff):
        self.classifier = classifier
        self._class_loss_coeff = class_loss_coeff

    def __call__(self, distribution, mc_samples=100):
        if isinstance(self.classifier, DDT):
            y_hat = self.classifier(distribution)
        elif isinstance(distribution, tfd.Deterministic):
            y_hat = self.classifier(distribution.loc)
        else:
            y_hat = tfp.monte_carlo.expectation(
                f=snt.BatchApply(self.classifier),
                samples=distribution.sample(mc_samples),
                log_prob=snt.Sequential(
                    [distribution.log_prob, lambda x: tf.expand_dims(x, -1)]),
                use_reparameterization=(
                    tfd.ReparameterizationType == tfd.FULLY_REPARAMETERIZED))
        return y_hat

    def forward_loss(self, distribution, labels, num_classes=10):
        y_hat = self(distribution)
        loss = tf.keras.losses.hinge(y_true=tf.one_hot(labels, num_classes),
                                     y_pred=y_hat)
        return {
            'class_loss': self._class_loss_coeff * tf.reduce_mean(loss),
            'y_hat': y_hat
        }
