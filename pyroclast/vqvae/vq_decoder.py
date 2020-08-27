import tensorflow as tf
import sonnet as snt

from .util import residual_block


class Decoder(tf.Module):
    """
    An abstract VQ-VAE decoder, which takes a stack of
    (differently-sized) input Tensors and produces a
    predicted output Tensor.
    Sub-classes should overload the forward() method.
    """

    def __call__(self, inputs):
        """
        Apply the decoder to a list of inputs.
        Args:
            inputs: a sequence of input Tensors. There may
              be more than one in the case of a hierarchy,
              in which case the top levels come first.
        Returns:
            A decoded Tensor.
        """
        raise NotImplementedError


class HalfDecoder(Decoder):
    """
    A decoder that upsamples by a factor of 2 in both
    dimensions.
    """

    def __init__(self, channels, out_channels):
        super().__init__()
        self.residual1 = residual_block(channels)
        self.residual2 = residual_block(channels)
        self.conv = snt.Conv2DTranspose(out_channels, 4, stride=2)

    def __call__(self, inputs):
        assert len(inputs) == 1
        x = inputs[0]
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        x = tf.nn.relu(x)
        x = self.conv(x)
        return x
