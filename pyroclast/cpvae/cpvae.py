import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from pyroclast.cpvae.ddt import transductive_box_inference, get_decision_tree_boundaries
import sklearn.tree
import tensorflow_datasets as tfds

from pyroclast.common.tf_util import img_discretized_logistic_log_prob


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
        super(CpVAE, self).__init__(name=name)
        self.encoder = encoder
        self.decoder = decoder
        self.decision_tree = decision_tree

        # tree_stuff
        self.lower = None
        self.upper = None
        self.values = None

        # multi-gaussian prior
        self.default_prior = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(latent_dimension),
            scale_diag=tf.ones(latent_dimension))
        self.class_priors = [
            tfp.distributions.MultivariateNormalDiag(
                loc=tf.Variable(np.zeros(latent_dimension, dtype=np.float32),
                                name='class_{}_loc'.format(i)),
                scale_diag=tfp.util.DeferredTensor(
                    tf.math.softplus,
                    tf.Variable(np.zeros(latent_dimension, dtype=np.float32),
                                name='class_{}_scale_diag'.format(i))))
            for i in range(class_num)
        ]

        # set distortion_fn
        if output_dist == 'disc_logistic':
            self.output_dist_scale = None  # deferred initialization
            self.distortion_fn = lambda x, x_hat, x_hat_scale: -img_discretized_logistic_log_prob(
                x_hat, x, x_hat_scale) / 100.
        elif output_dist == 'l2':
            self.distortion_fn = lambda x, x_hat: 500. * tf.reduce_mean(
                tf.square(x - x_hat), axis=[1, 2, 3])
        else:
            print('DISTORTION_FN NOT PROPERLY SPECIFIED')
            exit()

    def __call__(self, x, y=None):
        loc, scale_diag = self._encode(x)
        z_posterior = tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=scale_diag)
        z = z_posterior.sample()
        x_hat, x_hat_scale = self._decode(z)
        if not all(tf.math.is_finite(scale_diag)):
            print("SCALE DIAG ISN'T FINITE")
        y_hat = transductive_box_inference(loc, scale_diag, self.lower,
                                           self.upper, self.values)
        if not all(tf.math.is_finite(y_hat)):
            print("Y_HAT ISN'T FINITE")
        return x_hat, y_hat, z_posterior, x_hat_scale

    def _encode(self, x):
        loc, scale_diag = self.encoder(x)
        return loc, scale_diag

    def _decode(self, z):
        return self.decoder(z)

    def sample(self, sample_shape=(1), z=None):
        if z is None:
            z = self.default_prior.sample(sample_shape)
        return self._decode(z)

    def vae_loss(self, x, x_hat, x_hat_scale, z_posterior, y=None, epoch=None):
        # distortion
        if self.output_dist_scale is None:
            self.output_dist_scale = tfp.util.DeferredTensor(
                tf.math.softplus,
                tf.Variable(tf.ones(x_hat.shape[-3:]),
                            constraint=lambda x: tf.maximum(x, -3.),
                            name='LogScale'))
        distortion = self.distortion_fn(x, x_hat)

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
        if epoch is not None and epoch <= 10:
            rate = rate / (12 - epoch)
        return distortion, rate
