import tensorflow as tf

from pyroclast.common.models import get_network_builder


class VAEEncoder(tf.keras.Model):
    """Network parameterized encoder which outputs the parameters of a loc/scale distribution"""

    def __init__(self, network_name, name='enc'):
        super(VAEEncoder, self).__init__(name=name)
        self.net = get_network_builder(network_name)()
        self.loc = tf.keras.layers.Conv2D(1, 1, name='encoder_loc')
        self.scale = tf.keras.layers.Conv2D(1, 1, name='encoder_scale_diag_raw')

    def call(self, x):
        embed = self.net(x)
        inv_softplus_scale = self.scale(embed)
        loc = self.loc(embed)
        scale = tf.nn.softplus(inv_softplus_scale) + 1e-6
        return loc, scale


class VAEDecoder(tf.keras.Model):
    """Network parameterized decoder which outputs the parameters of a loc/scale distribution

    Currently, this class assumes that the data is an image of some kind.
    """

    def __init__(self, network_name, output_channels, name='dec'):
        super(VAEDecoder, self).__init__(name=name)
        self.net = get_network_builder(network_name)()
        self.loc = tf.keras.layers.Conv2D(output_channels,
                                          1,
                                          padding="same",
                                          name='decoder_loc')
        self.inv_softplus_scale = tf.keras.layers.Conv2D(
            output_channels,
            1,
            padding="same",
            name='decoder_inv_softplus_scale')

    def call(self, z):
        latent = self.net(z)
        loc, scale = self.loc(latent), tf.nn.softplus(
            self.inv_softplus_scale(latent)) + 1e-6
        return loc, scale
