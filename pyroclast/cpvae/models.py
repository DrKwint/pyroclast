import tensorflow as tf
import sonnet as snt


def build_classifier(network, num_classes, **network_kwargs):
    if isinstance(network, str):
        from pyroclast.common.models import get_network_builder
        network = get_network_builder(network)(**network_kwargs)

    return snt.Sequential([
        lambda x: network(x, **network_kwargs),
        snt.BatchFlatten(),
        snt.Linear(num_classes)
    ])


def build_encoder(network, latent_dim, **network_kwargs):
    if isinstance(network, str):
        from pyroclast.common.models import get_network_builder
        network = get_network_builder(network)(**network_kwargs)

    def encoder_builder(x):
        latent = network(x, **network_kwargs)
        latent = snt.BatchFlatten()(latent)
        loc = snt.Linear(latent_dim)(latent)
        scale = snt.Linear(latent_dim)(latent)
        return loc, scale

    return encoder_builder


def build_decoder(network, num_channels=3, **network_kwargs):
    if isinstance(network, str):
        from pyroclast.common.models import get_network_builder
        network = get_network_builder(network)(**network_kwargs)

    def decoder_builder(z):
        # TODO: remove the linear layer
        # `n` is local parameter for size of internal linear layer
        n = 32
        latent = snt.Linear(n)(z)
        latent = tf.reshape(latent, [-1, 1, 1, n])
        output = network(latent, **network_kwargs)
        output = snt.Conv2D(num_channels, 1, padding=snt.SAME)(output)
        return output

    return decoder_builder
