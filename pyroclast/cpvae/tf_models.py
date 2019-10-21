import tensorflow as tf
from pyroclast.common.models import get_network_builder


class Encoder(tf.keras.Model):

    def __init__(self, network_name, latent_dim, name='enc'):
        super(Encoder, self).__init__(name=name)
        self.net = get_network_builder(network_name)({})
        self.loc = tf.keras.layers.Dense(latent_dim, name='encoder_loc')
        self.scale = tf.keras.layers.Dense(latent_dim,
                                           name='encoder_scale_diag_raw')

    def call(self, x):
        embed = tf.reshape(self.net(x), [x.shape[0], -1])
        return self.loc(embed), tf.nn.softplus(self.scale(embed)) + 1e-6


class Decoder(tf.keras.Model):

    def __init__(self, network_name, image_size, name='dec'):
        super(Decoder, self).__init__(name=name)
        self.net = get_network_builder(network_name)({'image_size': image_size})
        self.final = tf.keras.layers.Conv2D(3, 3, padding="same")

    def call(self, z):
        latent = self.net(z)
        output = self.final(latent)
        return tf.nn.tanh(output)
