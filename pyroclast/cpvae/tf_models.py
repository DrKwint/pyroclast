import tensorflow as tf
from pyroclast.common.models import get_network_builder


class Encoder(tf.keras.Model):

    def __init__(self, network_name, latent_dim, name='enc'):
        super(Encoder, self).__init__(name=name)
        self.net = get_network_builder(network_name)()
        self.loc = tf.keras.layers.Dense(latent_dim, name='encoder_loc')
        self.scale = tf.keras.layers.Dense(latent_dim,
                                           name='encoder_scale_diag_raw')

    def call(self, x):
        embed = tf.reshape(self.net(x), [x.shape[0], -1])
        inv_softplus_scale = self.scale(embed)
        if not tf.reduce_all(tf.math.is_finite(inv_softplus_scale)):
            print("inv_softplus_scale encoder ISN'T FINITE")
        scale = tf.nn.softplus(inv_softplus_scale) + 1e-6
        if not tf.reduce_all(tf.math.is_finite(scale)):
            print("scale encoder ISN'T FINITE")
        return self.loc(embed), scale


class Decoder(tf.keras.Model):

    def __init__(self, network_name, image_size, name='dec'):
        super(Decoder, self).__init__(name=name)
        self.net = get_network_builder(network_name)()
        self.loc = tf.keras.layers.Conv2D(3, 3, padding="same")
        self.log_scale = tf.keras.layers.Conv2D(3, 3, padding="same")

    def call(self, z):
        latent = self.net(z)
        return self.loc(latent), self.log_scale(latent)
