import tensorflow as tf
import tensorflow_probability as tfp


class CpVAE(tf.Module):
    def __init__(self, encoder, decoder, classifier, name='cpvae'):
        super(CpVAE, self).__init__(name=name)
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.z_prior = None

    def __call__(self, x):
        loc, scale_diag = self.encoder(x)
        z_posterior = tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=tf.nn.softplus(scale_diag))
        if self.z_prior is None:
            self.z_prior = tfp.distributions.MultivariateNormalDiag(
                tf.zeros(z_posterior.event_shape),
                tf.ones(z_posterior.event_shape))
        z = z_posterior.sample()
        x_hat_raw = self.decoder(z)
        _, h, w, _ = x.shape.as_list()
        x_hat = tf.image.resize_with_crop_or_pad(x_hat_raw, h, w)
        y_hat = self.classifier(loc)
        return x_hat, y_hat, z_posterior

    def sample(self, z=None):
        if z is None:
            z = self.z_prior.sample()
        return self.decoder(z)

    def vae_loss(self, x, x_hat, z_posterior):
        distortion = tf.losses.mean_squared_error(labels=x, predictions=x_hat)
        rate = tfp.distributions.kl_divergence(z_posterior, self.z_prior)
        return distortion, rate
