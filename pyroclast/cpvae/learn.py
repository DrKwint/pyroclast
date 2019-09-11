import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from pyroclast.common.tf_util import run_epoch_ops, calculate_accuracy
from pyroclast.cpvae.cpvae import CpVAE
from pyroclast.cpvae.models import build_encoder, build_decoder, build_classifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def setup_celeba_data(batch_size):
    import tensorflow_datasets as tfds
    # load data
    data_dict, info = tfds.load('celeb_a', with_info=True)
    data_dict['train_bpe'] = info.splits['train'].num_examples // batch_size
    data_dict['test_bpe'] = info.splits['test'].num_examples // batch_size
    data_dict['shape'] = info.features['image'].shape

    data_dict['train'] = data_dict['train'].shuffle(1024).batch(
        batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    data_dict['test'] = data_dict['test'].batch(batch_size).prefetch(
        tf.data.experimental.AUTOTUNE)
    return data_dict


def train(session,
          train_data_iterator,
          train_batches_per_epoch,
          test_data_iterator,
          test_batches_per_epoch,
          epochs,
          train_op,
          train_verbose_dict,
          test_verbose_dict,
          verbose=True):
    for epoch in range(epochs):
        # re-initialize data iterators
        session.run(train_data_iterator.initializer)
        session.run(test_data_iterator.initializer)

        # run a training epoch
        print("Epoch", epoch)
        print("TRAIN")
        train_vals_dict = run_epoch_ops(
            session,
            train_batches_per_epoch,
            train_verbose_dict, [train_op],
            verbose=verbose)
        print({'mean ' + k: np.mean(v) for k, v in train_vals_dict.items()})

        # run a test epoch
        print("TEST")
        test_vals_dict = run_epoch_ops(
            session,
            test_batches_per_epoch,
            test_verbose_dict,
            verbose=verbose)
        print({'mean ' + k: np.mean(v) for k, v in test_vals_dict.items()})


def learn(data_dict,
          seed=None,
          latent_dim=32,
          epochs=100,
          batch_size=32,
          learning_rate=1e-3,
          tb_dir='./tb/'):
    del seed  # currently unused
    num_classes = data_dict['num_classes']

    # CELEB_A
    data_dict = setup_celeba_data(batch_size)

    # setup model
    encoder = build_encoder("conv_only", latent_dim)
    decoder = build_decoder("upscale_conv")
    classifier = build_classifier("mlp", num_classes)
    model = CpVAE(encoder, decoder, classifier)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # tensorboard
    global_step = tf.compat.v1.train.get_or_create_global_step()
    writer = tf.contrib.summary.create_file_writer(tb_dir)
    writer.set_as_default()

    # training loop
    for epoch in range(epochs):
        print("TRAIN")
        for batch in tqdm(data_dict['train'], total=data_dict['train_bpe']):
            global_step.assign_add(1)
            x = tf.cast(batch['image'], tf.float32) / 255.
            # label = batch['label']
            # train module
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
            loss = model.vae_loss(x, x_hat, z_posterior)
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
