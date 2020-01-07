import numpy as np
import sklearn.tree
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

from pyroclast.common.tf_util import img_discretized_logistic_log_prob
from pyroclast.cpvae.ddt import (get_decision_tree_boundaries,
                                 transductive_box_inference)


class CpVAE(tf.Module):

    def __init__(self,
                 encoder,
                 decoder,
                 decision_tree,
                 latent_dimension,
                 class_num,
                 box_num,
                 output_dist,
                 name='cpvae'):
        """Builds a CpVAE TF Module which performs generation and classification

        Args:
            encoder (Module): function from input data to loc and scale tensors each of length equal to `latent_dimension`
            decoder (Module): function from latent variable to loc and scale tensors of shape equal to the input data
            decision_tree (sklearn.tree.DecisionTreeClassifier):
            latent_dimension (int): number of dimensions in the latent variable
            class_num (int): number of classes to classify data into
            box_num (int): maximum number of boxes in the decision tree
            output_dist (str): name of the distribution to use as the VAE output, mostly used to calculate distortion
        """
        super(CpVAE, self).__init__(name=name)
        self.encoder = encoder
        self.decoder = decoder
        self.decision_tree = decision_tree

        # tree_stuff, these must be set before classification is attempted
        self.lower = None
        self.upper = None
        self.values = None

        # Set a default prior of the standard (0,I) Gaussian
        self.default_prior = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(latent_dimension),
            scale_diag=tf.ones(latent_dimension))
        # If the prior is per-class, use a Gaussian with learned parameters for each
        self.class_priors = [
            tfp.distributions.MultivariateNormalDiag(
                loc=tf.Variable(np.zeros(latent_dimension, dtype=np.float32),
                                name='class_{}_loc'.format(i)),
                scale_diag=tfp.util.DeferredTensor(
                    tf.math.softplus,
                    tf.Variable(np.ones(latent_dimension, dtype=np.float32),
                                name='class_{}_scale_diag'.format(i))))
            for i in range(class_num)
        ]

        # set distortion_fn
        # can be any fn which takes (data, output_mean, output_scale) and returns a value per datum
        if output_dist == 'disc_logistic':
            self.distortion_fn = lambda x, x_hat, x_hat_scale: -img_discretized_logistic_log_prob(
                x_hat, x, x_hat_scale)
        elif output_dist == 'l2':
            self.distortion_fn = lambda x, x_hat, x_hat_scale: tf.reduce_sum(
                tf.square(x - x_hat), axis=[1, 2, 3])
        elif output_dist == 'bernoulli':
            self.distortion_fn = lambda x, x_hat, x_hat_scale: -1. * tfp.distributions.Independent(
                tfp.distributions.Bernoulli(logits=x_hat), 3).log_prob(x)
        elif output_dist == 'continuous_bernoulli':
            self.distortion_fn = lambda x, x_hat, x_hat_scale: 0.
        else:
            print('DISTORTION_FN NOT PROPERLY SPECIFIED')
            exit()

    def __call__(self, x, y=None):
        # autoencode
        loc, scale_diag = self._encode(x)
        z_posterior = tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=scale_diag)
        z = z_posterior.sample()
        x_hat, x_hat_scale = self._decode(z)
        # classification
        assert not (self.lower is None or self.upper is None or
                    self.values is None)
        y_hat = transductive_box_inference(loc, scale_diag, self.lower,
                                           self.upper, self.values)
        return x_hat, y_hat, z_posterior, x_hat_scale

    def _encode(self, x):
        loc, scale_diag = self.encoder(x)
        return loc, scale_diag

    def _decode(self, z):
        return self.decoder(z)

    def sample(self, sample_num=1, use_class_prior=False):
        """Sample from the generative distribution

        Args:
            sample_shape (int): Number of samples
            use_class_prior (bool):
        """
        if use_class_prior:
            class_choices = tfp.distributions.Categorical(
                [1 / self.class_num] * self.class_num).sample(sample_num)
            z = np.array([self.class_priors[c].sample() for c in class_choices])
        else:
            z = self.default_prior.sample(sample_num)
        return self._decode(z)

    def vae_loss(self, x, x_hat, x_hat_scale, z_posterior, y=None):
        # distortion
        distortion = self.distortion_fn(x, x_hat, x_hat_scale)

        # rate
        if y is not None:
            if len(y.shape) == 1:
                y = tf.one_hot(y, len(self.class_priors))
            class_divergences = tf.stack([
                tfp.distributions.kl_divergence(z_posterior, prior)
                for prior in self.class_priors
            ],
                                         axis=1)  # batch_size x class_num
            rate = tf.reduce_sum(y * class_divergences, axis=1)
        else:
            rate = tfp.distributions.kl_divergence(z_posterior,
                                                   self.default_prior)
        return distortion, rate
