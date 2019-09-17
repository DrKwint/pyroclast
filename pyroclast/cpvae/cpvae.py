import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from pyroclast.cpvae.ddt import transductive_box_inference, get_decision_tree_boundaries
import sklearn.tree
import tensorflow_datasets as tfds

from pyroclast.common.tf_util import DiscretizedLogistic


class CpVAE(tf.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 decision_tree,
                 img_height,
                 img_width,
                 latent_dimension,
                 class_num,
                 box_num,
                 name='cpvae'):
        super(CpVAE, self).__init__(name=name)
        self.encoder = encoder
        self.decoder = decoder
        self.decision_tree = decision_tree
        self.img_height = img_height
        self.img_width = img_width
        self.z_prior = None

        # tree_stuff
        self.lower = None
        self.upper = None
        self.values = None
        """
        self._lower = tf.Variable(
            np.empty([box_num, latent_dimension], dtype=np.float32),
            trainable=False)
        self._upper = tf.Variable(
            np.empty([box_num, latent_dimension], dtype=np.float32),
            trainable=False)
        self._values = tf.Variable(
            np.empty([box_num, class_num], dtype=np.float32), trainable=False)
        """

    def __call__(self, x):
        loc, scale_diag = self._encode(x)
        z_posterior = tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=scale_diag)
        z = z_posterior.sample()
        x_hat = self._decode(z)
        y_hat = transductive_box_inference(loc, scale_diag, self.lower,
                                           self.upper, self.values)

        if self.z_prior is None:
            self.z_prior = tfp.distributions.MultivariateNormalDiag(
                tf.zeros(z_posterior.event_shape),
                tf.ones(z_posterior.event_shape))
        return x_hat, y_hat, z_posterior

    def _encode(self, x):
        loc, scale_diag = self.encoder(x)
        return loc, scale_diag

    def _decode(self, z):
        return tf.image.resize_with_crop_or_pad(
            self.decoder(z), self.img_height, self.img_width)

    def sample(self, sample_shape=(1), z=None):
        if z is None:
            z = self.z_prior.sample(sample_shape)
        return self._decode(z)

    def vae_loss(self, x, x_hat, z_posterior):
        #output_distribution = tfp.distributions.Independent(
        #    DiscretizedLogistic(x_hat), reinterpreted_batch_ndims=3)
        #distortion = -output_distribution.log_prob(x)
        distortion = tf.losses.mean_squared_error(labels=x, predictions=x_hat)
        rate = tfp.distributions.kl_divergence(z_posterior, self.z_prior)
        return distortion, rate
