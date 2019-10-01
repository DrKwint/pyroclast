import tensorflow as tf


def build_encoder(network, latent_dim, **network_kwargs):
    if isinstance(network, str):
        from pyroclast.common.models import get_network_builder
        network = get_network_builder(network)(**network_kwargs)

    def encoder_builder(x):
        latent = network(x, **network_kwargs)
        batch_size = x.get_shape().as_list()[0]
        latent = tf.reshape(latent, [batch_size, -1])
        loc = tf.keras.layers.Dense(latent_dim)(latent)
        scale = tf.keras.layers.Dense(latent_dim)(latent)
        return loc, scale

    return encoder_builder


def build_decoder(network, num_channels=3, **network_kwargs):
    if isinstance(network, str):
        from pyroclast.common.models import get_network_builder
        network = get_network_builder(network)(**network_kwargs)

    def decoder_builder(z):
        # TODO: remove the linear layer
        # `n` is local parameter for size of internal linear layer
        n, d = z.shape.as_list()
        latent = tf.reshape(z, [n, 8, 8, 1])
        output = network(latent, **network_kwargs)
        output = tf.keras.layers.Conv2D(num_channels, 1, padding="same")(output)
        return output  #tf.nn.tanh(output)

    return decoder_builder
