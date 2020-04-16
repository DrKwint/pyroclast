import numpy as np
import sklearn.tree
import tensorflow as tf
import tensorflow_probability as tfp

import tensorflow_datasets as tfds

tfd = tfp.distributions
tfb = tfp.bijectors

made = tfb.AutoregressiveNetwork(params=2,
                                 event_shape=latent_dimension,
                                 hidden_units=[20, 20],
                                 activation=tf.nn.relu,
                                 input_order='random')


class CpVAE(tf.Module):
    """Classifier + VAE model which is both generative and discriminative"""

    def __init__(self,
                 encoder,
                 decoder,
                 classifier,
                 latent_dimension,
                 output_dist,
                 is_class_prior,
                 class_num=None,
                 name='cpvae'):
        """Builds a CpVAE TF Module which performs generation and classification

        Args:
            encoder (Module): function from input data to loc and scale tensors each of length equal to `latent_dimension`
            decoder (Module): function from latent variable to loc and scale tensors of shape equal to the input data
            classifier (Module):
            latent_dimension (int): number of dimensions in the latent variable
            class_num (int): number of classes to classify data into
            box_num (int): maximum number of boxes in the decision tree
            output_dist (str): name of the distribution to use as the VAE output, mostly used to calculate distortion
        """
        super(CpVAE, self).__init__(name=name)
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.class_num = class_num
        self.is_class_prior = is_class_prior

        if is_class_prior:
            # If the prior is per-class, use a Gaussian with learned parameters for each
            self.class_priors = [
                tfp.distributions.MultivariateNormalDiag(
                    loc=tf.Variable(np.zeros(latent_dimension,
                                             dtype=np.float32),
                                    name='class_{}_loc'.format(i)),
                    scale_diag=tfp.util.DeferredTensor(
                        tf.Variable(np.ones(latent_dimension, dtype=np.float32),
                                    name='class_{}_scale_diag'.format(i)),
                        tf.math.softplus)) for i in range(class_num)
            ]
        else:
            # Set a default prior of the standard (0,I) Gaussian
            self.default_prior = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(latent_dimension),
                scale_diag=tf.ones(latent_dimension))

        # set distortion_fn
        # can be any fn which takes (data, output_mean, output_scale) and returns a value per datum
        # TODO: move this logic elsewhere and change CpVAE to take actual distributions
        if output_dist == 'disc_logistic':
            self.distortion_fn = lambda x, x_hat, x_hat_scale: -tfp.distributions.Independent(
                tfd.QuantizedDistribution(
                    distribution=tfd.TransformedDistribution(
                        distribution=tfd.Logistic(loc=x_hat, scale=x_hat_scale),
                        bijector=tfb.Shift(-0.5 / 256)),
                    low=0.,
                    high=1.), 3).log_prob(x)
            self.output_fn = lambda x_hat, x_hat_scale: x_hat
        elif output_dist == 'l2':
            self.distortion_fn = lambda x, x_hat, x_hat_scale: tf.reduce_sum(
                tf.square(x - x_hat), axis=[1, 2, 3])
            self.output_fn = lambda x_hat, x_hat_scale: x_hat
        elif output_dist == 'bernoulli':
            self.distortion_fn = lambda x, x_hat, x_hat_scale: -tfp.distributions.Independent(
                tfp.distributions.Bernoulli(logits=x_hat), 3).log_prob(x)
            self.output_fn = lambda x_hat, x_hat_scale: x_hat
        else:
            print('DISTORTION_FN NOT PROPERLY SPECIFIED')
            exit()

    def __call__(self, x):
        # autoencode
        z_loc, z_scale_diag = self._encode(x)
        z_posterior = self.posterior(z_loc, scale_diag=z_scale_diag)
        z = z_posterior.sample()
        # classification
        y_hat = self.classifier(z_loc, z_scale_diag)
        return z_posterior, y_hat

    def posterior(self, x, scale_diag=None):
        # x is data if scale diag is None, otherwise it's loc
        if scale_diag is None:
            loc, scale_diag = self._encode(x)
        else:
            loc = x
        z_posterior = tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=scale_diag)
        return z_posterior

    def encode(self, x):
        loc, scale_diag = self.encoder(x)
        z_posterior = self.posterior_fn(loc, scale_diag)
        return z_posterior

    def _decode(self, z):
        return self.decoder(z)

    def autoencode(self, x):
        loc, scale_diag = self._encode(x)
        z_posterior = tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=scale_diag)
        z = z_posterior.sample()
        x_hat_loc, x_hat_log_scale = self._decode(z)
        return self.output_fn(x_hat_loc, x_hat_log_scale)

    def sample(self, sample_num=1):
        """Sample from the generative distribution

        Args:
            sample_shape (int): Number of samples
            use_class_prior (bool):
        """
        if self.is_class_prior:
            class_choices = tfp.distributions.Categorical(
                [1 / self.class_num] * self.class_num).sample(sample_num)
            z = np.array([self.class_priors[c].sample() for c in class_choices])
        else:
            z = self.default_prior.sample(sample_num)
        x_hat_loc, x_hat_scale = self._decode(z)
        return self.output_fn(x_hat_loc, x_hat_scale)

    def vae_loss(self, x, x_hat, x_hat_scale, z_posterior, y=None):
        # distortion
        distortion = self.distortion_fn(x, x_hat, x_hat_scale)

        # rate
        if self.is_class_prior:
            assert y is not None
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
