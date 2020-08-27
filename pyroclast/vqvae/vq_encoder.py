import tensorflow as tf
import sonnet as snt

from .util import residual_block


class Encoder(tf.Module):
    """
    An abstract VQ-VAE encoder, which takes input Tensors,
    shrinks them, and quantizes the result.
    Sub-classes should implement the encode() method.
    Args:
        num_channels: the number of channels in the latent
          codebook.
        num_latents: the number of entries in the latent
          codebook.
        kwargs: arguments to pass to the VQ layer.
    """

    def __init__(self,
                 embedding_dim,
                 num_embeddings,
                 vq_commitment_cost,
                 vq_decay=0.99):
        super().__init__()
        self._vq = snt.nets.VectorQuantizerEMA(embedding_dim, num_embeddings,
                                               vq_commitment_cost, vq_decay)

    def encode(self, x):
        """
        Encode a Tensor before the VQ layer.
        Args:
            x: the input Tensor.
        Returns:
            A Tensor with the correct number of output
              channels (according to self.vq).
        """
        raise NotImplementedError

    def __call__(self, x, is_training):
        """
        Apply the encoder.
        See VQ.forward() for return values.
        """
        return [self._vq(self.encode(x), is_training)]


class TopBottomEncoder(object):

    def __init__(self, embed_net, up_net, top_encoder, bottom_encoder):
        """
        Args:
            embed_net: network from x -> A
            top_enc: encoder from A -> B
            up_net: net from B -> A
            bottom enc: encoder from A -> A

        where A is the size of the bottom representation
        and B is the size of the top representation
        """
        self._embed = embed_net
        self._up_net = up_net
        self._top_enc = top_encoder
        self._bottom_enc = bottom_encoder

    def __call__(self, inputs):
        x = inputs[0]
        h_embed = self._embed(x)
        e_top = self._top_enc(h_embed)
        he_top = self._up_net(e_top['quantize'])
        e_bottom = self._bottom_enc(tf.concat([h_embed, he_top]))
        return (e_top, e_bottom)


class HalfEncoder(Encoder):
    """
    An encoder that cuts the input size in half in both
    dimensions.
    """

    def __init__(self,
                 channels,
                 num_embeddings,
                 vq_commitment_cost,
                 vq_decay=0.99):
        super().__init__(channels, num_embeddings, vq_commitment_cost, vq_decay)
        self.down_conv = snt.Conv2D(channels, 3, stride=2)
        self.residual1 = residual_block(channels)
        self.residual2 = residual_block(channels)

    def encode(self, x):
        x = tf.nn.relu(self.down_conv(x))
        x = x + self.residual1(x)
        x = x + self.residual2(x)
        return x
