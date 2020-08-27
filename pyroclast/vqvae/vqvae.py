from collections.abc import Iterable

import sonnet as snt
import tensorflow as tf

from .vq_decoder import HalfDecoder
from .vq_encoder import HalfEncoder


class VQVAE(tf.Module):

    def __init__(self, encoder, decoder):
        self._encoder = encoder
        self._decoder = decoder

    def __call__(self, inputs, is_training=False):
        vqs = self._encoder(inputs, is_training)
        x_hat = self._decoder([vq['quantize'] for vq in vqs])
        return {'vqs': vqs, 'recon': x_hat}

    def forward_loss(self, inputs):
        outputs = self(inputs, is_training=True)
        vq_loss = sum([vq['loss'] for vq in outputs['vqs']])

        x_recon = outputs['recon']
        recon_error = tf.reduce_mean(
            (x_recon - inputs)**2)  # / self._data_variance
        loss = recon_error + vq_loss
        outputs.update({
            'loss': loss,
            'recon_loss': recon_error,
            'vq_loss': vq_loss
        })
        return outputs
