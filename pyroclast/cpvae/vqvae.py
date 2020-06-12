import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

from pyroclast.cpvae.abstract_vae import AbstractVAE
from pyroclast.common.models import get_network_builder

tfd = tfp.distributions


class VQVAE(AbstractVAE):

    def __init__(self, vq_bottom, vq_top, embedding_dim, data_variance):
        super(VQVAE, self).__init__()
        self._top_encoder4 = get_network_builder("vqvae_cifar10_encoder")(
            downscale=4)
        self._top_encoder2 = get_network_builder("vqvae_cifar10_encoder")(
            downscale=2)
        self._bottom_encoder4 = get_network_builder("vqvae_cifar10_encoder")(
            downscale=4)
        self._top_decoder = get_network_builder("vqvae_cifar10_decoder")(
            upscale=8)
        self._bottom_decoder = get_network_builder("vqvae_cifar10_decoder")(
            upscale=4)
        self._vq_bottom = vq_bottom
        self._vq_top = vq_top
        self._data_variance = data_variance
        self._pre_vq_conv1 = snt.Conv2D(output_channels=embedding_dim,
                                        kernel_shape=(1, 1),
                                        stride=(1, 1),
                                        name="to_vq")
        self._pre_vq_conv2 = snt.Conv2D(output_channels=embedding_dim,
                                        kernel_shape=(1, 1),
                                        stride=(1, 1),
                                        name="to_vq2")

    def __call__(self, inputs, is_training=False):
        z_bottom = self._bottom_encoder4(inputs)
        vq_output_bottom = self._vq_bottom(self._pre_vq_conv1(z_bottom),
                                           is_training=is_training)
        z_top = self._top_encoder2(
            tf.concat(
                [self._top_encoder4(inputs), vq_output_bottom['quantize']], -1))
        vq_output_top = self._vq_top(self._pre_vq_conv2(z_top),
                                     is_training=is_training)
        x_recon = self._bottom_decoder(vq_output_bottom['quantize'])
        x_recon += self._top_decoder(vq_output_top['quantize'])
        outputs = {
            'z_sample': z_top,
            'z': tfd.Deterministic(z_top),
            'x_recon': x_recon,
            'vq_output_bottom': vq_output_bottom,
            'vq_output_top': vq_output_top,
        }
        return outputs

    def forward_loss(self, inputs):
        outputs = self(inputs, is_training=True)
        x_recon = outputs['x_recon']
        if 'vq_output_top' in outputs:
            vq_loss = outputs['vq_output_top']['loss'] + outputs[
                'vq_output_bottom']['loss']
        else:
            vq_loss = outputs['vq_output']['loss']

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
