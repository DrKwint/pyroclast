import os

import numpy as np
import sklearn
import tensorflow as tf

from pyroclast.common.util import ensure_dir_exists
from pyroclast.cpvae.model import CpVAE
from pyroclast.cpvae.tf_models import VAEDecoder, VAEEncoder
from pyroclast.cpvae.ddt import DDT


def build_saveable_objects(optimizer_name, encoder_name, decoder_name,
                           learning_rate, num_classes, num_channels, latent_dim,
                           output_dist, max_tree_depth, model_dir, model_name):
    # model
    encoder = VAEEncoder(encoder_name, latent_dim)
    decoder = VAEDecoder(decoder_name, num_channels)
    ddt = DDT(max_tree_depth)
    model = CpVAE(encoder,
                  decoder,
                  ddt,
                  latent_dimension=latent_dim,
                  class_num=num_classes,
                  output_dist=output_dist,
                  is_class_prior=True)

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
        'ckpt_manager': ckpt_manager
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
