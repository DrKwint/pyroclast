import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class TreeVAE(tf.Module):

    def __init__(self,
                 encoder,
                 posterior_fn,
                 decoder,
                 classifier,
                 prior,
                 output_distribution_fn,
                 class_priors=None,
                 use_analytic_classifier=False):
        self.encoder = encoder
        self.posterior_fn = posterior_fn
        self.decoder = decoder
        self.classifier = classifier
        self.output_distribution_fn = output_distribution_fn

        self.prior = prior
        self.class_priors = class_priors
        self.use_analytic_classifier = use_analytic_classifier

    def __call__(self, x):
        z_posterior = self.posterior(x)
        if self.use_analytic_classifier:
            leaf_probs, y_hat = self.classifier.classify_analytic(
                z_posterior.parameters['loc'],
                z_posterior.parameters['scale_diag'])
        else:
            leaf_probs, y_hat = self.classifier.classify_numerical(z_posterior)
        return z_posterior, leaf_probs, y_hat

    def vae_loss(self, x, z_posterior, y=None, training=True):
        z_sample = z_posterior.sample()
        loc, scale = self.decoder(z_sample)

        # calculate distortion
        if isinstance(self.output_distribution_fn, tfd.PixelCNN):
            distortion = self.output_distribution_fn.log_prob(
                x, conditional_input=loc, training=training)
        else:
            distortion = self.output_distribution_fn(loc, scale).log_prob(x)

        # calculate rate
        # use implmeneted KL if available
        if isinstance(self.prior,
                      tfd.MultivariateNormalLinearOperator) and isinstance(
                          z_posterior, tfd.MultivariateNormalLinearOperator):
            if self.class_priors is not None and y is not None:  # if using class prior
                if len(y.shape) == 1:
                    y = tf.one_hot(y, len(self.class_priors))
                    class_divergences = tf.stack(
                        [
                            tfp.distributions.kl_divergence(z_posterior, prior)
                            for prior in self.class_priors
                        ],
                        axis=1)  # batch_size x class_num
                rate = tf.reduce_sum(y * class_divergences, axis=1)
            else:  # not using class prior
                rate = tfp.distributions.kl_divergence(z_posterior, self.prior)
        else:  # otherwise, use numerical estimate
            if self.class_priors is not None and y is not None:  # if using class prior
                rate = [
                    tfp.vi.monte_carlo_variational_loss(class_prior.log_prob,
                                                        z_posterior,
                                                        sample_size=100)
                    for class_prior in self.class_priors
                ]
            else:  # not using class prior
                rate = tfp.vi.monte_carlo_variational_loss(self.prior.log_prob,
                                                           z_posterior,
                                                           sample_size=100)
        return -distortion, rate

    def posterior(self, x):
        loc, scale_diag = self.encoder(x)
        z_posterior = self.posterior_fn(loc, scale_diag)
        return z_posterior

    def encode(self, x):
        return self.encoder(x)[0]

    def decode(self, z):
        return self.decoder(z)[0]

    def sample_prior(self, is_class=False, class_=None):
        if is_class:
            if class_ is None:
                class_ = tf.argmax(
                    tfd.Categorical(logits=1. /
                                    float(len(self.class_priors))).sample())
            z_sample = self.class_priors[class_].sample()
        else:
            z_sample = self.prior.sample(1)
        return self.decode(z_sample)

    def sample_posterior(self, x):
        z_posterior = self.posterior(x)
        z_sample = z_posterior.sample()
        return self.decode(z_sample)

    def decoder_trainable_variables(self):
        var_list = self.decoder.trainable_variables
        if isinstance(self.output_distribution_fn, tfd.PixelCNN):
            var_list += self.output_distribution_fn.trainable_variables
        return var_list
