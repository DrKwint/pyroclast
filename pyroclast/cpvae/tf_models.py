import tensorflow as tf
from pyroclast.common.models import get_network_builder


class Encoder(tf.keras.Model):
    def __init__(self, latent_dim, name='enc'):
        super(Encoder, self).__init__(name=name)
        self.net = get_network_builder('conv_only')({})
        self.loc = tf.keras.layers.Dense(latent_dim)
        self.scale = tf.keras.layers.Dense(latent_dim)

    def call(self, x):
        n, _, _, _ = x.shape
        embed = self.net(x)
        embed = tf.reshape(embed, [n, -1])
        return self.loc(embed), self.scale(embed)


class Decoder(tf.keras.Model):
    def __init__(self, name='dec'):
        super(Decoder, self).__init__(name=name)
        self.net = get_network_builder('upscale_conv')()
        self.final = tf.keras.layers.Conv2D(3, 1, padding="same")

    def call(self, z):
        n, d = z.shape.as_list()
        latent = tf.reshape(z, [n, 8, 8, 1])
        output = self.net(latent)
        output = self.final(output)
        return tf.nn.tanh(output)
