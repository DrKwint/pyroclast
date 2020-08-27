import tensorflow as tf


def _make_residual(channels):
    return nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(channels, channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(channels, channels, 1),
    )


class VQVAE(tf.Module):

    def __init__(self, encoders, decoders, vqs):
        assert len(encoders) == len(decoders) == len(vqs)
        self._encoders = encoders
        self._decoders = decoders
        self._vqs = vqs

    def __call__(self, x, is_training=False):
