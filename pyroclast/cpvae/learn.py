import os

import numpy as np
import tensorflow as tf

from pyroclast.common.tf_util import calculate_accuracy, run_epoch_ops
from pyroclast.common.util import dummy_context_mgr, img_postprocess
from pyroclast.cpvae.util import build_model, update_model_tree

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def setup(data_dict, optimizer, encoder, decoder, learning_rate, latent_dim,
          output_dist, max_tree_depth, max_tree_leaf_nodes, load_dir,
          output_dir):
    num_classes = data_dict['num_classes']
    num_channels = data_dict['shape'][-1]

    # setup model vars
    model, optimizer, global_step = build_model(
        optimizer_name=optimizer,
        encoder_name=encoder,
        decoder_name=decoder,
        learning_rate=learning_rate,
        num_classes=num_classes,
        num_channels=num_channels,
        latent_dim=latent_dim,
        output_dist=output_dist,
        max_tree_depth=max_tree_depth,
        max_tree_leaf_nodes=max_tree_leaf_nodes)

    #checkpointing and tensorboard
    writer = tf.summary.create_file_writer(output_dir)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=model,
                                     global_step=global_step)
    ckpt_manager = tf.train.CheckpointManager(checkpoint,
                                              directory=os.path.join(
                                                  output_dir, 'model'),
                                              max_to_keep=3,
                                              keep_checkpoint_every_n_hours=2)

    # load trained model, if available
    if load_dir:
        status = checkpoint.restore(tf.train.latest_checkpoint(str(load_dir)))
        print("load: ", status.assert_existing_objects_matched())

    # train a ddt
    update_model_tree(data_dict['train'],
                      model,
                      epoch='visualize',
                      num_classes=num_classes,
                      output_dir=output_dir)
    return model, optimizer, global_step, writer, ckpt_manager


def learn(
        data_dict,
        encoder,
        decoder,
        seed=None,
        latent_dim=128,
        epochs=1000,
        max_tree_depth=6,
        max_tree_leaf_nodes=32,
        tree_update_period=3,
        optimizer='rmsprop',  # adam or rmsprop
        learning_rate=1e-4,
        output_dist='l2',  # disc_logistic or l2 or hybrid
        output_dir='./',
        load_dir=None,
        num_samples=5,
        clip_norm=0.,
        alpha=1e-2,
        beta=5.,
        gamma=5.,
        silent=False):
    model, optimizer, global_step, writer, ckpt_manager = setup(
        data_dict, optimizer, encoder, decoder, learning_rate, latent_dim,
        output_dist, max_tree_depth, max_tree_leaf_nodes, load_dir, output_dir)

    # define minibatch fn
    def run_minibatch(epoch, batch, is_train=True):
        x = tf.cast(batch['image'], tf.float32) / 255.
        labels = tf.cast(batch['label'], tf.int32)

        with tf.GradientTape() if is_train else dummy_context_mgr() as tape:
            global_step.assign_add(1)
            x_hat, y_hat, z_posterior, x_hat_scale = model(x)
            print(x_hat)
            y_hat = tf.cast(y_hat, tf.float32)  # from double to single fp
            distortion, rate = model.vae_loss(x,
                                              x_hat,
                                              x_hat_scale,
                                              z_posterior,
                                              y=labels)
            classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=y_hat)
            loss = tf.reduce_mean(alpha * distortion + beta * rate +
                                  gamma * classification_loss)

        # calculate gradients for current loss
        if is_train:
            gradients = tape.gradient(loss, model.trainable_variables)
            if clip_norm:
                clipped_gradients, pre_clip_global_norm = tf.clip_by_global_norm(
                    gradients, clip_norm)
            else:
                clipped_gradients = gradients
            optimizer.apply_gradients(
                zip(clipped_gradients, model.trainable_variables))

        prefix = 'train ' if is_train else 'validate '
        with writer.as_default():
            prediction = tf.math.argmax(y_hat, axis=1, output_type=tf.int32)
            classification_rate = tf.reduce_mean(
                tf.cast(tf.equal(prediction, labels), tf.float32))
            tf.summary.scalar(prefix + "loss/mean distortion",
                              alpha * tf.reduce_mean(distortion),
                              step=global_step)
            tf.summary.scalar(prefix + "loss/mean rate",
                              beta * tf.reduce_mean(rate),
                              step=global_step)
            tf.summary.scalar(prefix + "loss/mean classification_loss",
                              gamma * tf.reduce_mean(classification_loss),
                              step=global_step)
            tf.summary.scalar(prefix + "classification_rate",
                              classification_rate,
                              step=global_step)
            tf.summary.scalar(prefix + "loss/total loss",
                              loss,
                              step=global_step)
            tf.summary.scalar(prefix + 'posterior/mean stddev',
                              tf.reduce_mean(z_posterior.stddev()),
                              step=global_step)
            tf.summary.scalar(prefix + 'posterior/min stddev',
                              tf.reduce_min(z_posterior.stddev()),
                              step=global_step)
            tf.summary.scalar(prefix + 'posterior/max stddev',
                              tf.reduce_max(z_posterior.stddev()),
                              step=global_step)

            if is_train:
                for (v, g) in zip(model.trainable_variables, gradients):
                    if g is None:
                        continue
                    tf.summary.scalar(prefix +
                                      'gradient/mean of {}'.format(v.name),
                                      tf.reduce_mean(g),
                                      step=global_step)
                if clip_norm:
                    tf.summary.scalar("gradient/global norm",
                                      pre_clip_global_norm,
                                      step=global_step)

    # run training loop
    for epoch in range(epochs):
        # train
        for batch in data_dict['train']:
            run_minibatch(epoch, batch, is_train=True)

        # test
        for batch in data_dict['test']:
            run_minibatch(epoch, batch, is_train=False)

        # save and update
        ckpt_manager.save(checkpoint_number=epoch)
        if epoch % tree_update_period == 0:
            update_model_tree(data_dict['train'], model, epoch,
                              data_dict['num_classes'], output_dir)

        # sample
        for i in range(num_samples):
            im = np.squeeze(model.sample()[0])
            if output_dist == 'bernoulli':
                im = np.exp(im)
            im.save(
                os.path.join(output_dir,
                             "epoch_{}_sample_{}.png".format(epoch, i)))
