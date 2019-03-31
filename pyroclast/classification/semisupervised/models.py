import sonnet as snt
import tensorflow as tf


def build_classifier(network, **network_kwargs):
    if isinstance(network, str):
        from pyroclast.common.models import get_network_builder
        network = get_network_builder(network)(**network_kwargs)

    def classifier_builder(x, class_num):
        latent = network(x)
        logits = snt.Linear(class_num)(latent)
        return logits

    return classifier_builder


def build_encoder(network, **network_kwargs):
    if isinstance(network, str):
        from pyroclast.common.models import get_network_builder
        network = get_network_builder(network)(**network_kwargs)

    def encoder_builder(x, y, latent_dim):
        # TODO: combine x and y to be network input
        latent = network(x)
        loc = snt.Linear(latent_dim)(latent)
        scale = snt.Linear(latent_dim)(latent)
        return loc, scale

    return encoder_builder


def build_decoder(network, **network_kwargs):
    if isinstance(network, str):
        from pyroclast.common.models import get_network_builder
        network = get_network_builder(network)(**network_kwargs)

    def decoder_builder(y, z):
        # TODO: combine y and z to be network input
        return network(z)

    return decoder_builder


# TODO: add probabilities to this so they can be built cleanly too
