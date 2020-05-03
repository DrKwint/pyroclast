import functools

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from pyroclast.cpvae.ddt import DDT
from pyroclast.cpvae.distributions import get_distribution_builder
from pyroclast.cpvae.model import TreeVAE
from pyroclast.cpvae.tf_models import VAEDecoder, VAEEncoder


def build_saveable_objects(optimizer_name, encoder_name, decoder_name,
                           learning_rate, num_channels, latent_dim, prior_name,
                           posterior_name, output_distribution_name,
                           max_tree_depth, model_dir, model_name):
    # model
    encoder = VAEEncoder(encoder_name, latent_dim)
    decoder = VAEDecoder(decoder_name, num_channels)
    ddt = DDT(max_tree_depth, use_analytic=False)
    classifier = tf.keras.layers.Dense(10)
    if prior_name == 'iaf_prior':
        prior_ar_network = tfp.bijectors.AutoregressiveNetwork(
            params=2,
            hidden_units=[512, 512, 512],
            activation='elu',
            name='prior_ar_network')
        prior = get_distribution_builder(prior_name)(latent_dim,
                                                     prior_ar_network)
    else:
        prior = get_distribution_builder(prior_name)(latent_dim)
    posterior_fn = get_distribution_builder(posterior_name)()
    if posterior_name == 'iaf_posterior':
        ar_network = tfp.bijectors.AutoregressiveNetwork(
            params=2,
            hidden_units=[512, 512, 512],
            activation='elu',
            name='posterior_ar_network')
        posterior_fn = functools.partial(posterior_fn, ar_network=ar_network)
    else:
        ar_network = None
    output_distribution_fn = get_distribution_builder(
        output_distribution_name)()
    model = TreeVAE(encoder=encoder,
                    posterior_fn=posterior_fn,
                    decoder=decoder,
                    classifier=classifier,
                    prior=prior,
                    output_distribution_fn=output_distribution_fn)
    if prior_name == 'iaf_prior':
        model.prior_ar_network = prior_ar_network
    if posterior_name == 'iaf_posterior':
        model.posterior_ar_network = ar_network

    # optimizer
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                             beta_1=0.5,
                                             epsilon=0.01)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate)
    else:
        print("OPTIMIMIZER NOT PROPERLY SPECIFIED")
        exit()

    # global_step
    global_step = tf.compat.v1.train.get_or_create_global_step()

    # checkpoint
    save_dict = {
        model_name + '_optimizer': optimizer,
        model_name + '_model': model,
        model_name + '_global_step': global_step
    }
    checkpoint = tf.train.Checkpoint(**save_dict)

    # checkpoint manager
    ckpt_manager = tf.train.CheckpointManager(checkpoint,
                                              directory=model_dir,
                                              max_to_keep=3)

    return {
        'model': model,
        'optimizer': optimizer,
        'global_step': global_step,
        'checkpoint': checkpoint,
        'ckpt_manager': ckpt_manager,
        'classifier': ddt
    }


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
