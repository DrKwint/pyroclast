import tensorflow_probability as tfp

from pyroclast.cpvae.abstract_vae import AbstractVAE
import tensorflow as tf
import sonnet as snt

tfd = tfp.distributions


class VAE(AbstractVAE):

    def __init__(self,
                 encoder,
                 decoder,
                 prior,
                 posterior_fn,
                 output_distribution_fn,
                 latent_channels,
                 data_channels,
                 beta=1.):
        self._encoder = encoder
        self._decoder = decoder
        self._prior = prior
        self._posterior_fn = posterior_fn
        self._output_fn = output_distribution_fn
        self._beta = beta

        self._posterior_loc = snt.Linear(latent_channels)
        self._posterior_scale = snt.Sequential(
            [snt.Linear(latent_channels), tf.nn.softplus])
        self._output_loc = snt.Conv2D(data_channels, 1)
        self._output_scale = snt.Sequential(
            [snt.Conv2D(data_channels, 1), tf.nn.softplus])

    def __call__(self, x):
        z_posterior = self.posterior(x)
        z_sample = z_posterior.sample()
        output_distribution = self._output_distribution(z_sample)
        return {
            'z': z_posterior,
            'z_sample': z_sample,
            'x': output_distribution
        }

    def forward_loss(self, inputs, mc_samples=100):
        outputs = self(inputs)
        if isinstance(outputs['x'], tfd.PixelCNN):
            loc = self._output_loc(self._decoder(outputs['z_sample']))
            recon_loss = -1 * outputs['x'].log_prob(inputs,
                                                    conditional_input=loc)
        elif 'Quantized' in outputs['x'].name:
            recon_loss = -1 * outputs['x'].log_prob((inputs + 0.5) * 255)
        else:
            recon_loss = -1 * outputs['x'].log_prob(inputs)

        z_posterior = outputs['z']
        if isinstance(self._prior,
                      tfd.MultivariateNormalLinearOperator) and isinstance(
                          z_posterior, tfd.MultivariateNormalLinearOperator):
            latent_loss = tfd.kl_divergence(z_posterior, self._prior)
        else:
            latent_loss = tfp.vi.monte_carlo_variational_loss(
                self._prior.log_prob, z_posterior, sample_size=mc_samples)

        loss = tf.reduce_mean(recon_loss + (self._beta * latent_loss))
        outputs.update({
            'gen_loss': loss,
            'recon_loss': recon_loss,
            'latent_loss': latent_loss
        })
        return outputs

    def posterior(self, inputs):
        encoder_embed = self._encoder(inputs)
        loc, scale_diag = self._posterior_loc(
            encoder_embed), self._posterior_scale(encoder_embed)
        return tfd.Independent(self._posterior_fn(loc, scale_diag),
                               len(loc.shape) - 2)

    def _output_distribution(self, z_sample):
        # Assumes that data is NHWC and that the decoder outputs the correct NHW dims
        if isinstance(self._output_fn, tfd.PixelCNN):
            return self._output_fn
        decoder_embed = self._decoder(z_sample)
        loc, scale = self._output_loc(decoder_embed), self._output_scale(
            decoder_embed)
        return tfp.distributions.Independent(self._output_fn(loc, scale), 3)

    def output_distribution(self, inputs):
        posterior = self.posterior(inputs)
        posterior_sample = posterior.sample()
        return self._output_distribution(posterior_sample)

    def output_point_estimate(self, inputs):
        dist = self.output_distribution(inputs)
        if 'Quantized' in dist.name:
            return (dist.sample() / 255.) - 0.5
        else:
            return dist.sample()
        """
        posterior = self.posterior(inputs)
        posterior_sample = posterior.sample()
        loc = self._output_loc(self._decoder(posterior_sample))
        tf.print("output_loc", tf.reduce_min(loc), tf.reduce_mean(loc),
                 tf.reduce_max(loc))
        if isinstance(self._output_fn, tfd.PixelCNN):
            loc = self._output_fn.sample(conditional_input=loc)
            print(loc.shape)
            exit()
        return loc
        """
