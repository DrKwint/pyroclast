import tensorflow as tf

from pyroclast.common.models import get_network_builder


class VAEEncoder(tf.keras.Model):
    """Network parameterized encoder which outputs the parameters of a loc/scale distribution"""

    def __init__(self, network_name, latent_dim, name='enc'):
        super(VAEEncoder, self).__init__(name=name)
        self.net = get_network_builder(network_name)()
        self.loc = tf.keras.layers.Dense(latent_dim, name='encoder_loc')
        self.scale = tf.keras.layers.Dense(latent_dim,
                                           name='encoder_scale_diag_raw')
        self.flatten = tf.keras.layers.Flatten()

    def call(self, x):
        embed = self.flatten(self.net(x))
        inv_softplus_scale = self.scale(embed)
        scale = tf.nn.softplus(inv_softplus_scale) + 1e-6
        return self.loc(embed), scale


class VAEDecoder(tf.keras.Model):
    """Network parameterized decoder which outputs the parameters of a loc/scale distribution

    Currently, this class assumes that the data is an image of some kind.
    """

    def __init__(self, network_name, output_channels, name='dec'):
        super(VAEDecoder, self).__init__(name=name)
        self.net = get_network_builder(network_name)()
        self.loc = tf.keras.layers.Conv2D(
            output_channels,
            3,
            padding="same",
            #activation=tf.nn.relu,
            name='decoder_loc')
        self.inv_softplus_scale = tf.keras.layers.Conv2D(
            output_channels,
            3,
            padding="same",
            name='decoder_inv_softplus_scale')

    def call(self, z):
        latent = self.net(z)
        loc, scale = self.loc(latent), tf.nn.softplus(
            self.inv_softplus_scale(latent)) + 1e-6
        return loc, scale
