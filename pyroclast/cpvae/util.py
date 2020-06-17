import functools

import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

from pyroclast.common.models import get_network_builder
from pyroclast.cpvae.distributions import get_distribution_builder
from pyroclast.cpvae.vae import VAE
from pyroclast.cpvae.vqvae import VQVAE

tfd = tfp.distributions


def build_vqvae(encoder_name,
                decoder_name,
                layers,
                data_channels,
                data_variance,
                embedding_dim,
                num_embeddings,
                commitment_cost,
                decay=0.99,
                vq_use_ema=True,
                output_channels=3):

    if vq_use_ema:
        vector_quantizer = snt.nets.VectorQuantizerEMA(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
            decay=decay)
        vector_quantizer_top = snt.nets.VectorQuantizerEMA(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
            decay=decay)
    else:
        vector_quantizer = snt.nets.VectorQuantizer(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost)
        vector_quantizer_top = snt.nets.VectorQuantizer(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost)
    if layers == 1:
        vector_quantizer_top = None

    model = VQVAE(encoder_name,
                  decoder_name,
                  vector_quantizer,
                  vq_top=vector_quantizer_top,
                  embedding_dim=embedding_dim,
                  data_variance=data_variance,
                  output_channels=output_channels)
    return model


def build_vae(encoder_name, decoder_name, prior_name, posterior_name,
              output_distribution_name, latent_dim, data_channels, beta):
    encoder = get_network_builder(encoder_name)()
    decoder = get_network_builder(decoder_name)()
    ### distributions
    # prior
    if 'ar_prior' in prior_name:
        prior_ar_network = tfp.bijectors.AutoregressiveNetwork(
            params=2,
            hidden_units=[64, 64, 64],
            activation='elu',
            name='prior_ar_network')
        prior = get_distribution_builder(prior_name)(
            latent_dim, prior_ar_network
        )  # note, this needs to be called with an ar network which needs to be changed
    else:
        prior = get_distribution_builder(prior_name)(latent_dim)

    # posterior
    posterior_fn = get_distribution_builder(posterior_name)()
    if 'ar_posterior' in posterior_name:
        posterior_ar_network = tfp.bijectors.AutoregressiveNetwork(
            params=2,
            hidden_units=[256, 256],
            activation='elu',
            name='posterior_ar_network')
        posterior_fn = functools.partial(posterior_fn,
                                         ar_network=posterior_ar_network)
    else:
        posterior_ar_network = None

    # output_distribution
    output_distribution_fn = get_distribution_builder(
        output_distribution_name)()

    model = VAE(encoder, decoder, prior, posterior_fn, output_distribution_fn,
                latent_dim, data_channels, beta)
    return model


def build_checkpoint(save_dict, dir_):
    checkpoint = tf.train.Checkpoint(**save_dict)

    # checkpoint manager
    ckpt_manager = tf.train.CheckpointManager(checkpoint,
                                              directory=dir_,
                                              max_to_keep=3)
    return checkpoint, ckpt_manager


def build_saveable_objects(optimizer_name, learning_rate, objects,
                           model_save_dir):
    # optimizer
    if optimizer_name == 'adam':
        optimizer = snt.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate)
    else:
        print("OPTIMIMIZER NOT AVAILABLE")
        exit()

    # global_step
    global_step = tf.compat.v1.train.get_or_create_global_step()

    # checkpoint
    objects['global_step'] = global_step
    checkpoint, ckpt_manager = build_checkpoint(objects, model_save_dir)

    return {
        'optimizer': optimizer,
        'global_step': global_step,
        'checkpoint': checkpoint,
        'ckpt_manager': ckpt_manager,
    }


def load_args_from_dir(dir_):
    from pathlib import Path
    import json
    with open(Path(dir_) / 'parameters.json') as param_json:
        loaded_args = json.load(param_json)
    args = loaded_args['args']
    args.update(loaded_args['module_kwargs'])
    return args


def calculate_latent_params_by_class(labels, loc, scale_diag, class_num,
                                     latent_dimension):
    # update class stats
    if len(labels.shape) > 1:
        labels = np.argmax(labels, axis=1)
    class_locs = np.zeros([class_num, latent_dimension])
    class_scales = np.zeros([class_num, latent_dimension])
    sum_sq = tf.square(scale_diag) + tf.square(loc)
    for l in range(class_num):
        class_locs[l] = np.mean(tf.gather(loc, tf.where(tf.equal(labels, l))))
        class_scales[l] = np.mean(tf.gather(sum_sq, tf.where(tf.equal(
            labels, l))),
                                  axis=0) - np.square(class_locs[l])
    return class_locs, class_scales


def calculate_walk(origin, destination, steps=8, dim=None):
    steps = tf.expand_dims(
        tf.cast(tf.concat([tf.range(0., 1., delta=1. / float(steps)), [1.]], 0),
                tf.float64), 1)
    delta = destination - origin
    if dim is None:
        return origin + (delta * steps)
    else:
        delta = delta * tf.one_hot(dim, delta.shape[-1], dtype=tf.float64)
        return origin + (delta * steps), destination - (delta * steps)
