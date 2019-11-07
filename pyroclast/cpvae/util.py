import os

import sklearn
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import numpy as np

from pyroclast.cpvae.cpvae import CpVAE
from pyroclast.cpvae.tf_models import Decoder, Encoder
from tqdm import tqdm
from pyroclast.cpvae.ddt import transductive_box_inference, get_decision_tree_boundaries


def build_model(optimizer_name, learning_rate, num_classes, latent_dim,
                image_size, max_tree_depth, max_tree_leaf_nodes):
    tfvar_objs = dict()

    # model
    encoder = Encoder('celeba_enc', latent_dim)
    decoder = Decoder('celeba_dec', image_size)
    decision_tree = sklearn.tree.DecisionTreeClassifier(
        max_depth=max_tree_depth,
        min_weight_fraction_leaf=0.01,
        max_leaf_nodes=max_tree_leaf_nodes)
    model = CpVAE(encoder,
                  decoder,
                  decision_tree,
                  latent_dimension=latent_dim,
                  class_num=num_classes,
                  box_num=max_tree_leaf_nodes)

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

    return model, optimizer, global_step


def fit_and_calculate_dt_boxes(decision_tree, z, label, class_num,
                               latent_dimension):
    # train ensemble
    decision_tree.fit(z, label)
    lower_, upper_, values_ = get_decision_tree_boundaries(
        decision_tree, latent_dimension, class_num)
    return lower_, upper_, values_


def calculate_latent_params_by_class(labels, loc, scale_diag, class_num,
                                     latent_dimension):
    # update class stats
    if len(labels.shape) > 1: labels = np.argmax(labels, axis=1)
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


def calculate_celeba_latent_values(ds, model, label_attr, limit=None):
    locs = []
    scales = []
    samples = []
    attrs = []
    for i, batch in tqdm(enumerate(ds)):
        if limit is not None:
            if i > limit:
                break
        loc, scale_diag = model._encode(
            tf.dtypes.cast(batch['image'], tf.float32))
        locs.append(loc)
        scales.append(scale_diag)
        z_posterior = tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=scale_diag)
        z = z_posterior.sample()
        samples.append(z)
        attrs.append(batch['attributes'][label_attr])
    locs = tf.concat(locs, 0)
    scales = tf.concat(scales, 0)
    samples = tf.concat(samples, 0)
    attrs = tf.concat(attrs, 0)
    return locs, scales, samples, attrs


def update_model_tree(ds, model, epoch, label_attr, output_dir, limit=None):
    locs, scales, samples, labels = calculate_celeba_latent_values(ds,
                                                                   model,
                                                                   label_attr,
                                                                   limit=limit)
    labels = tf.cast(labels, tf.int32)
    lower_, upper_, values_ = fit_and_calculate_dt_boxes(
        model.decision_tree, samples, labels, 2, samples.shape[-1])
    model.lower = lower_
    model.upper = upper_
    model.values = values_
    class_locs, class_scales = calculate_latent_params_by_class(
        labels, locs, scales, 2, samples.shape[-1])
    sklearn.tree.export_graphviz(model.decision_tree,
                                 out_file=os.path.join(
                                     output_dir,
                                     'ddt_epoch{}.dot'.format(epoch)),
                                 filled=True,
                                 rounded=True)
    return class_locs, class_scales
