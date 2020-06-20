import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

from pyroclast.cpvae.abstract_vae import AbstractVAE
from pyroclast.common.models import get_network_builder

tfd = tfp.distributions


class VQVAE(AbstractVAE):

    def __init__(self,
                 encoder_name,
                 decoder_name,
                 vq_bottom,
                 embedding_dim,
                 data_variance,
                 vq_top=None,
                 output_channels=3):
        super(VQVAE, self).__init__()
        if vq_top is not None:
            self._top_encoder4 = get_network_builder(encoder_name)(downscale=4)
            self._top_encoder2 = get_network_builder(encoder_name)(downscale=2)
            self._top_decoder = get_network_builder(decoder_name)(
                upscale=8, output_channels=output_channels)
            self._pre_vq_conv2 = snt.Conv2D(output_channels=embedding_dim,
                                            kernel_shape=(1, 1),
                                            stride=1,
                                            name="to_vq2")
        self._bottom_encoder4 = get_network_builder(encoder_name)(downscale=4)
        self._bottom_decoder = get_network_builder(decoder_name)(
            upscale=4, output_channels=output_channels)
        self._vq_bottom = vq_bottom
        self._vq_top = vq_top
        self._data_variance = data_variance
        self._pre_vq_conv1 = snt.Conv2D(output_channels=embedding_dim,
                                        kernel_shape=(1, 1),
                                        stride=1,
                                        name="to_vq")
        self.prior = None

    @property
    def num_layers(self):
        return self._vq_top is not None

    def __call__(self, inputs, is_training=False):
        z_bottom = self._bottom_encoder4(inputs)
        vq_output_bottom = self._vq_bottom(self._pre_vq_conv1(z_bottom),
                                           is_training=is_training)
        x_recon = self._bottom_decoder(vq_output_bottom['quantize'])
        outputs = {}
        if self._vq_top is not None:
            z_top = self._top_encoder2(
                tf.concat(
                    [self._top_encoder4(inputs), vq_output_bottom['quantize']],
                    -1))
            vq_output_top = self._vq_top(self._pre_vq_conv2(z_top),
                                         is_training=is_training)
            x_recon += self._top_decoder(vq_output_top['quantize'])
            outputs = {
                'z_sample': z_top,
                'z': tfd.Deterministic(z_top),
                'z_sample_top': z_top,
                'z_sample_bottom': z_bottom,
                'x_recon': x_recon,
                'vq_output_bottom': vq_output_bottom,
                'vq_output_top': vq_output_top,
            }
        else:
            outputs = {
                'z_sample': z_bottom,
                'z': tfd.Deterministic(z_bottom),
                'x_recon': x_recon,
                'vq_output_bottom': vq_output_bottom,
            }
        return outputs

    def forward_loss(self, inputs):
        outputs = self(inputs, is_training=True)
        x_recon = outputs['x_recon']
        if 'vq_output_top' in outputs:
            vq_loss = outputs['vq_output_top']['loss'] + outputs[
                'vq_output_bottom']['loss']
        else:
            vq_loss = outputs['vq_output_bottom']['loss']

        recon_error = tf.reduce_mean(
            (x_recon - inputs)**2) / self._data_variance
        loss = recon_error + vq_loss
        outputs.update({
            'gen_loss': loss,
            'recon_loss': recon_error,
            'vq_loss': vq_loss
        })
        return outputs

    def posterior(self, inputs):
        pass

    def output_distribution(self, inputs):
        outputs = self(inputs, is_training=False)
        return tfd.Deterministic(outputs['x_recon'])

    def output_point_estimate(self, inputs):
        return self.output_distribution(inputs).loc

    def recon_from_codes(self, codes):
        prior_z_bottom = codes[-1]
        vq_output_bottom = self._vq_bottom(prior_z_bottom, is_training=False)
        x_recon = self._bottom_decoder(vq_output_bottom['quantize'])
        return x_recon


class AuxiliaryPrior(tf.Module):

    def __init__(self, code_shape, num_embeddings, vqvae=None):
        self.parent_vqvae = vqvae
        self.pcnn = tfp.distributions.PixelCNN(image_shape=code_shape,
                                               num_resnet=1,
                                               num_hierarchies=2,
                                               num_filters=32,
                                               num_logistic_mix=5,
                                               dropout_p=.3,
                                               high=num_embeddings - 1)

    def __call__(self):
        pass

    def forward_loss(self, inputs):
        log_prob = self.pcnn.log_prob(inputs)
        return {'gen_loss': -tf.reduce_mean(log_prob)}

    def output_point_estimate(self, inputs):
        code = self.pcnn.sample()
        code = code[:7, :7]
        code = tf.expand_dims(code, 0)
        return self.parent_vqvae.recon_from_codes([code])
