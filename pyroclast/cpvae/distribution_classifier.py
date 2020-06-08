import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class DistributionClassifier(tf.Module):

    def __init__(self, classifier):
        self.classifier = classifier

    def call(self, distribution, mc_samples=100):
        if isinstance(self.classifier, DDT):
            y_hat = self.classifier(distribution)
        else:
            y_hat = tfp.monte_carlo.expectation(
                f=snt.BatchApply(self.classifier),
                samples=distribution.sample(mc_samples),
                log_prob=distribution.log_prob,
                use_reparameterization=(
                    tfd.reparameterization_type == tfd.FULLY_REPARAMETERIZED))
        return y_hat

    def forward_loss(self, distribution, labels):
        y_hat = self(distribution)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                              logits=y_hat)
