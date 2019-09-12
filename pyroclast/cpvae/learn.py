import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from pyroclast.common.tf_util import run_epoch_ops, calculate_accuracy
from pyroclast.cpvae.cpvae import CpVAE
from pyroclast.cpvae.models import build_encoder, build_decoder
from PIL import Image
import sklearn.tree
import tensorflow_datasets as tfds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def setup_celeba_data(batch_size):
    # load data
    data_dict, info = tfds.load('celeb_a', with_info=True)
    data_dict['train_bpe'] = info.splits['train'].num_examples // batch_size
    data_dict['test_bpe'] = info.splits['test'].num_examples // batch_size
    data_dict['shape'] = info.features['image'].shape

    data_dict['all_train'] = data_dict['train']
    data_dict['train'] = data_dict['train'].shuffle(1024).batch(
        batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    data_dict['all_test'] = data_dict['test']
    data_dict['test'] = data_dict['test'].batch(batch_size).prefetch(
        tf.data.experimental.AUTOTUNE)
    return data_dict


def learn(data_dict,
          seed=None,
          latent_dim=32,
          epochs=100,
          batch_size=32,
          learning_rate=1e-3,
          max_tree_depth=5,
          max_tree_leaf_nodes=16,
          tb_dir='./tb/'):
    del seed  # currently unused
    num_classes = data_dict['num_classes']

    # CELEB_A
    data_dict = setup_celeba_data(batch_size)

    # setup model
    encoder = build_encoder("conv_only", latent_dim)
    decoder = build_decoder("upscale_conv")
    decision_tree = sklearn.tree.DecisionTreeClassifier(
        max_depth=max_tree_depth,
        min_weight_fraction_leaf=0.01,
        max_leaf_nodes=max_tree_leaf_nodes)
    classifier = lambda z: decision_tree.predict(z)
    model = CpVAE(encoder, decoder, classifier, img_height=218, img_width=178, latent_dimension=latent_dim, class_num=num_classes, box_num=max_tree_leaf_nodes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # tensorboard
    global_step = tf.compat.v1.train.get_or_create_global_step()
    writer = tf.contrib.summary.create_file_writer(tb_dir)
    writer.set_as_default()

    # training loop
    for epoch in range(epochs):
        print("TRAIN")
        i = 0
        for i, batch in tqdm(enumerate(data_dict['train']), total=data_dict['train_bpe']):
            if i > 10:
                break
            global_step.assign_add(1)
            # move data from [0,255] to [-1,1]
            x = (tf.cast(batch['image'], tf.float32) - 128) / 128.
            # label = batch['label']
            
            with tf.GradientTape() as tape:
                x_hat, y_hat, z_posterior = model(x)
                distortion, rate = model.vae_loss(x, x_hat, z_posterior)
                loss = distortion + rate
            # calculate gradients for current loss
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))
            
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar(
                    "mean_train_loss", loss, family='loss')
                """
                accuracy = tf.reduce_mean(
                    tf.cast(
                        tf.equal(tf.argmax(y_hat, axis=1), label), tf.float32))
                tf.contrib.summary.scalar(
                    "mean_train_accuracy", accuracy, family='accuracy')
                """
        print("TEST")
        for batch in tqdm(data_dict['test'], total=data_dict['test_bpe']):
            x = tf.cast(batch['image'], tf.float32) / 255.
            # label = batch['label']
            x_hat, y_hat, z_posterior = model(x)
            distortion, rate = model.vae_loss(x, x_hat, z_posterior)
            loss = distortion + rate
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar(
                    "mean_test_loss", loss, family='loss')
                """
                accuracy = tf.reduce_mean(
                    tf.cast(
                        tf.equal(tf.argmax(y_hat, axis=1), label), tf.float32))
                tf.contrib.summary.scalar(
                    "mean_test_accuracy", accuracy, family='accuracy')
                """

        sample = ((np.squeeze(model.sample()) * 128) + 128).astype(np.uint8)
        print(np.min(sample), np.max(sample))
        print(sample.shape)
        im = Image.fromarray(sample)
        im.save("epoch_{}_sample.png".format(epoch))