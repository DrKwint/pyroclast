import os

import numpy as np
import tensorflow as tf

from pyroclast.common.util import img_postprocess
from pyroclast.common.tf_util import calculate_accuracy, run_epoch_ops
from pyroclast.common.util import dummy_context_mgr
from pyroclast.cpvae.util import update_model_tree, build_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def setup(data_dict, optimizer, learning_rate, latent_dim, image_size,
          output_dist, max_tree_depth, max_tree_leaf_nodes, load_dir,
          output_dir, label_attr):
    num_classes = data_dict['num_classes']

    # setup model vars
    model, optimizer, global_step = build_model(
        optimizer_name=optimizer,
        learning_rate=learning_rate,
        num_classes=num_classes,
        latent_dim=latent_dim,
        image_size=image_size,
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
                      label_attr=label_attr,
                      output_dir=output_dir,
                      limit=10)
    return model, optimizer, global_step, writer, ckpt_manager


def learn(
        data_dict,
        seed=None,
        latent_dim=128,
        epochs=1000,
        image_size=128,
        max_tree_depth=5,
        max_tree_leaf_nodes=16,
        tree_update_period=10,
        label_attr='No_Beard',
        optimizer='adam',  # adam or rmsprop
        learning_rate=1e-3,
        classification_coeff=1.,
        output_dist='disc_logistic',  # disc_logistic or l2
        output_dir='./',
        load_dir=None,
        num_samples=5):
    model, optimizer, global_step, writer, ckpt_manager = setup(
        data_dict, optimizer, learning_rate, latent_dim, image_size,
        output_dist, max_tree_depth, max_tree_leaf_nodes, load_dir, output_dir,
        label_attr)

    # define minibatch fn
    def run_minibatch(epoch, batch, is_train=True):
        x = tf.cast(batch['image'], tf.float32)
        labels = tf.cast(batch['attributes'][label_attr], tf.int32)

        with tf.GradientTape() if is_train else dummy_context_mgr() as tape:
            x_hat, y_hat, z_posterior = model(x)
            y_hat = tf.cast(y_hat, tf.float32)  # from double to single fp
            distortion, rate = model.vae_loss(x, x_hat, z_posterior, y=labels)
            classification_loss = classification_coeff * tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=y_hat)
            loss = tf.reduce_mean(distortion + rate + classification_loss)

        # calculate gradients for current loss
        if is_train:
            global_step.assign_add(1)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        with writer.as_default():
            prediction = tf.math.argmax(y_hat, axis=1, output_type=tf.int32)
            classification_rate = tf.reduce_mean(
                tf.cast(tf.equal(prediction, labels), tf.float32))
            tf.summary.scalar("loss/distortion",
                              tf.reduce_mean(distortion),
                              step=global_step)
            tf.summary.scalar("loss/rate",
                              tf.reduce_mean(rate),
                              step=global_step)
            tf.summary.scalar("loss/classification_loss",
                              tf.reduce_mean(classification_loss),
                              step=global_step)
            tf.summary.scalar("classification_rate",
                              classification_rate,
                              step=global_step)
            tf.summary.scalar("loss/sum_loss", loss, step=global_step)
            global_norm = tf.math.sqrt(
                tf.reduce_sum([tf.norm(g)**2 for g in gradients]))
            tf.summary.scalar("gradient/global norm",
                              global_norm,
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
            update_model_tree(data_dict['train'], model, epoch, label_attr,
                              output_dir)

        # sample
        for i in range(num_samples):
            im = img_postprocess(np.squeeze(model.sample()))
            im.save(
                os.path.join(output_dir,
                             "epoch_{}_sample_{}.png".format(epoch, i)))
