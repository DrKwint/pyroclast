import functools

import numpy as np
import tensorflow as tf

from pyroclast.cpvae.ddt import DDT
from pyroclast.cpvae.distributions import get_distribution_builder
from pyroclast.cpvae.model import TreeVAE
from pyroclast.cpvae.tf_models import VAEDecoder, VAEEncoder

import tensorflow_probability as tfp


def build_saveable_objects(optimizer_name, encoder_name, decoder_name,
                           learning_rate, num_channels, latent_dim, prior_name,
                           posterior_name, output_distribution_name,
                           max_tree_depth, model_dir, model_name):
    # model
    encoder = VAEEncoder(encoder_name, latent_dim)
    decoder = VAEDecoder(decoder_name, num_channels)
    ddt = DDT(max_tree_depth)
    prior = get_distribution_builder(prior_name)(latent_dim)
    posterior_fn = get_distribution_builder(posterior_name)()
    if posterior_name == 'iaf_posterior':
        ar_network = tfp.bijectors.AutoregressiveNetwork(
            params=2,
            hidden_units=[32, 32],
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
                    classifier=ddt,
                    prior=prior,
                    output_distribution_fn=output_distribution_fn,
                    use_analytic_classifier=False)
    if posterior_name == 'iaf_posterior':
        model.ar_network = ar_network

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


def get_node_members(ds, model, node_id):

    def member_fn(batch):
        loc, _ = model._encode(tf.dtypes.cast(batch['image'], tf.float32))
        membership = model.decision_tree.decision_path(loc)
        return membership[node_id]

    return ds.filter(member_fn)
