import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors

GAUSSIAN_PRIOR = lambda latent_dimension: tfp.distributions.MultivariateNormalDiag(
    loc=tf.zeros(latent_dimension), scale_diag=tf.ones(latent_dimension))

LEARNED_GAUSSIAN_CLASS_PRIOR = lambda latent_dimension, class_num: [
    tfp.distributions.MultivariateNormalDiag(
        loc=tf.Variable(np.zeros(latent_dimension, dtype=np.float32),
                        name='class_{}_loc'.format(i)),
        scale_diag=tfp.util.DeferredTensor(
            tf.Variable(np.ones(latent_dimension, dtype=np.float32),
                        name='class_{}_scale_diag'.format(i)), tf.math.softplus
        )) for i in range(class_num)
]

MNIST_PIXELCNN = tfd.PixelCNN(
    image_shape=(28, 28, 1),
    num_resnet=1,
    num_hierarchies=2,
    num_filters=32,
    num_logistic_mix=5,
    dropout_p=.3,
)

DISCRETIZED_LOGISTIC = lambda loc, scale: tfp.distributions.Independent(
    tfd.QuantizedDistribution(distribution=tfd.TransformedDistribution(
        distribution=tfd.Logistic(loc, scale), bijector=tfb.Shift(-0.5 / 256)),
                              low=0.,
                              high=1.), 3)


class TreeVAE(tf.Module):

    def __init__(self,
                 encoder,
                 posterior_fn,
                 decoder,
                 classifier,
                 latent_dimension,
                 prior,
                 pixel_cnn=None,
                 class_priors=None):
        self.encoder = encoder
        self.posterior_fn = posterior_fn
        self.decoder = decoder
        self.pixel_cnn = pixel_cnn

        self.prior = prior
        self.class_priors = class_priors

    def __call__(self, x):
        z_posterior = self._encode(x)
        y_hat = self.classifier(z_posterior)
        return z_posterior, y_hat

    def vae_loss(self, x, z_posterior, y=None, training=True):
        z_sample = z_posterior.sample()

        # calculate distortion
        if type(self.output_distribution) is tfd.PixelCNN:
            distortion = self.pixel_cnn.log_prob(
                x, conditional_input=self.decoder(z_sample), training=training)
        else:
            distortion = self.output_distribution(
                self.decoder(z_sample)).log_prob(x)

        # calculate rate
        # use implmeneted KL if available
        if (type(self.prior),
                type(z_posterior)) in tfd.kullback_leibler._DIVERGENCES:
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
                rate = tfp.distributions.kl_divergence(z_posterior,
                                                       self.default_prior)
        else:  # otherwise, use numerical estimate
            if self.class_priors is not None and y is not None:  # if using class prior
                rate = [
                    tfp.vi.monte_carlo_variational_loss(class_prior,
                                                        z_posterior,
                                                        sample_size=10)
                    for class_prior in self.class_priors
                ]
            else:  # not using class prior
                rate = tfp.vi.monte_carlo_variational_loss(self.z_prior,
                                                           z_posterior,
                                                           sample_size=100)
        return distortion, rate

    def encode(self, x):
        loc, scale_diag = self.encoder(x)
        z_posterior = self.posterior_fn(loc, scale_diag)
        return z_posterior

    def decode(self, z):
        return self.pixel_cnn.sample(conditional_input=self.decoder(z))

    def sample_prior(self, is_class=False, class_=None):
        if is_class:
            if class_ is None:
                class_ = tf.argmax(
                    tfd.Categorical(logits=1. /
                                    float(len(self.class_priors))).sample())
            z_sample = self.class_priors[class_].sample()
        else:
            z_sample = self.prior.sample()
        return self.decode(z_sample)

    def sample_posterior(self, x):
        z_posterior = self.encode(x)
        z_sample = z_posterior.sample()
        return self.decode(z_sample)
