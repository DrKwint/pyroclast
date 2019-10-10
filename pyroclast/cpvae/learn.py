import os

import numpy as np
import sklearn.tree
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from PIL import Image
from tqdm import tqdm

from pyroclast.common.tf_util import calculate_accuracy, run_epoch_ops
from pyroclast.cpvae.cpvae import CpVAE
from pyroclast.cpvae.ddt import transductive_box_inference, get_decision_tree_boundaries
from pyroclast.cpvae.tf_models import Encoder, Decoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
CRANE = os.environ['HOME'] == "/home/scott/equint"
GAMMA = 1000.


def setup_celeba_data(batch_size):
    # load data
    data_dir = './data/' if CRANE else None
    data_dict, info = tfds.load('celeb_a', with_info=True, data_dir=data_dir)
    data_dict['train_bpe'] = info.splits['train'].num_examples // batch_size
    data_dict['test_bpe'] = info.splits['test'].num_examples // batch_size
    data_dict['shape'] = info.features['image'].shape

    data_dict['all_train'] = data_dict['train']
    data_dict['train'] = data_dict['train'].shuffle(1024).batch(batch_size)
    data_dict['all_test'] = data_dict['test']
    data_dict['test'] = data_dict['test'].batch(batch_size)
    return data_dict


def concat_dicts(list_of_dicts):
    concat_dict = dict()
    for key in list_of_dicts[0].keys():
        concat_dict[key] = tf.concat(values=[d[key] for d in list_of_dicts],
                                     axis=0)
    return concat_dict


def calculate_latent_values(ds, model, label_attr):
    locs = []
    scales = []
    samples = []
    attrs = []
    for batch in tqdm(ds):
        loc, scale_diag = model._encode(tf.to_float(batch['image']))
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


def update_model_tree(ds, model, epoch, label_attr, output_dir):
    locs, scales, samples, labels = calculate_latent_values(
        ds, model, label_attr)
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


def center_crop(x, crop_h, crop_w=None):
    # crop the images to [crop_h,crop_w,3] then resize to [resize_h,resize_w,3]
    if crop_w is None:
        crop_w = crop_h  # the width and height after cropped
    h, w = x.shape[1:3]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return x[:, j:j + crop_h, i:i + crop_w, :]


def learn(data_dict,
          seed=None,
          latent_dim=128,
          epochs=1000,
          batch_size=64,
          max_tree_depth=5,
          max_tree_leaf_nodes=16,
          label_attr='No_Beard',
          output_dir='./',
          load_dir=None):
    del seed  # currently unused
    num_classes = data_dict['num_classes']

    # CELEB_A
    data_dict = setup_celeba_data(batch_size)
    num_classes = 1

    # setup model
    encoder = Encoder('celeba_enc', 64)
    decoder = Decoder('celeba_dec')
    decision_tree = sklearn.tree.DecisionTreeClassifier(
        max_depth=max_tree_depth,
        min_weight_fraction_leaf=0.01,
        max_leaf_nodes=max_tree_leaf_nodes)
    model = CpVAE(encoder,
                  decoder,
                  decision_tree,
                  img_height=218,
                  img_width=178,
                  latent_dimension=latent_dim,
                  class_num=num_classes,
                  box_num=max_tree_leaf_nodes)
    optimizer = tf.keras.optimizers.RMSprop(1e-3)

    # tensorboard
    global_step = tf.compat.v1.train.get_or_create_global_step()
    writer = tf.contrib.summary.create_file_writer(output_dir)
    writer.set_as_default()
    
    # reload if data exists
    if load_dir:
        new_root = tf.train.Checkpoint(optimizer=optimizer, model=model)
        status = new_root.restore(tf.train.latest_checkpoint(str(load_dir)))
        print("load: ", status.assert_existing_objects_matched())

    # training loop
    update_model_tree(data_dict['train'],
                      model,
                      epoch='init',
                      label_attr=label_attr,
                      output_dir=output_dir)
    for epoch in range(epochs):
        print("TRAIN")
        for i, batch in tqdm(enumerate(data_dict['train']),
                             total=data_dict['train_bpe']):
            global_step.assign_add(1)
            # move data from [0,255] to [-1,1]
            x = (tf.cast(batch['image'], tf.float32) / 127.5) - 1
            x = center_crop(x)
            print(x.shape)
            exit()
            labels = tf.cast(batch['attributes'][label_attr], tf.int32)

            with tf.GradientTape() as tape:
                x_hat, y_hat, z_posterior = model(x)
                y_hat = tf.cast(y_hat, tf.float32)
                distortion, rate = model.vae_loss(x, x_hat, z_posterior)
                classification_loss = GAMMA * tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=y_hat)
                loss = tf.reduce_mean(distortion + rate + classification_loss)
            # calculate gradients for current loss
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar("distortion",
                                          distortion,
                                          family='train')
                tf.contrib.summary.scalar("rate", rate, family='train')
                tf.contrib.summary.scalar("classification_loss",
                                          classification_loss,
                                          family='train')
                tf.contrib.summary.scalar("sum_loss", loss, family='train')

        print("TEST")
        for batch in tqdm(data_dict['test'], total=data_dict['test_bpe']):
            x = (tf.cast(batch['image'], tf.float32) / 127.5) - 1
            labels = tf.cast(batch['attributes'][label_attr], tf.int32)

            x_hat, y_hat, z_posterior = model(x)
            y_hat = tf.cast(y_hat, tf.float32)
            distortion, rate = model.vae_loss(x, x_hat, z_posterior)
            classification_loss = GAMMA * tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=y_hat)
            loss = tf.reduce_mean(distortion + rate + classification_loss)

            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar("distortion",
                                          distortion,
                                          family='test')
                tf.contrib.summary.scalar("rate", rate, family='test')
                tf.contrib.summary.scalar("classification_loss",
                                          classification_loss,
                                          family='test')
                tf.contrib.summary.scalar("mean_test_loss", loss, family='test')

        print("SAVE CHECKPOINT")
        checkpoint_prefix = os.path.join(output_dir, "ckpt")
        root = tf.train.Checkpoint(optimizer=optimizer,
                                   model=model,
                                   step=global_step)

        root.save(checkpoint_prefix)

        print("UPDATE")
        update_model_tree(data_dict['train'], model, epoch, label_attr,
                          output_dir)

        print("SAMPLE")
        for i in range(5):
            sample = np.squeeze(model.sample())
            im = Image.fromarray(((sample + 1) * 127.5).astype('uint8'),
                                 mode='RGB')
            im.save("epoch_{}_sample_{}.png".format(epoch, i))
