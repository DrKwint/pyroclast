import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

from pyroclast.cpvae.abstract_vae import AbstractVAE

tfd = tfp.distributions


class VQVAE(AbstractVAE):

    def __init__(self,
                 encoder,
                 decoder,
                 vector_quantizer,
                 embedding_dim,
                 data_variance,
                 top_encoder=None,
                 top_decoder=None,
                 top_vector_quantizer=None):
        super(VQVAE, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._vq = vector_quantizer
        self._data_variance = data_variance
        self._pre_vq_conv1 = snt.Conv2D(output_channels=embedding_dim,
                                        kernel_shape=(1, 1),
                                        stride=(1, 1),
                                        name="to_vq")
        if top_encoder and top_decoder and top_vector_quantizer:
            self._encoder_top = top_encoder
            self._decoder_top = top_decoder
            self._vq_top = top_vector_quantizer
            self._pre_vq_conv2 = snt.Conv2D(output_channels=embedding_dim // 2,
                                            kernel_shape=(1, 1),
                                            stride=(1, 1),
                                            name="to_vq")

    def __call__(self, inputs, is_training=False):
        z_bottom = self._encoder(inputs)
        vq_output_bottom = self._vq(self._pre_vq_conv1(z_bottom),
                                    is_training=is_training)
        if self._vq_top:
            z_top = self._encoder_top(z_bottom)
            vq_output_top = self._vq_top(self._pre_vq_conv2(z_bottom),
                                         is_training=is_training)
        x_recon = self._decoder(vq_output_bottom['quantize'])
        return {
            'z_sample': z_bottom,
            'z': tfd.Deterministic(z_bottom),
            'x_recon': x_recon,
            'vq_output': vq_output_bottom,
        }

    def forward_loss(self, inputs):
        outputs = self(inputs, is_training=True)
        x_recon = outputs['x_recon']
        vq_output = outputs['vq_output']

        recon_error = tf.reduce_mean(
            (x_recon - inputs)**2) / self._data_variance
        loss = recon_error + vq_output['loss']
        outputs.update({
            'gen_loss': loss,
            'recon_loss': recon_error,
            'vq_loss': vq_output['loss']
        })
        return outputs

    def posterior(self, inputs):
        z = self._pre_vq_conv1(self._encoder(inputs))
        vq_output = self._vq(z, is_training=False)
        return tfd.Deterministic(vq_output['quantize'])

    def output_distribution(self, inputs):
        vq = self.posterior(inputs).loc
        x_recon = self._decoder(vq)
        return tfd.Deterministic(x_recon)

    def output_point_estimate(self, inputs):
        return self.output_distribution(inputs).loc
