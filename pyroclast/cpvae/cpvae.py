import functools
import os
import os.path as osp

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from pyroclast.common.early_stopping import EarlyStopping
from pyroclast.common.util import dummy_context_mgr
from pyroclast.cpvae.ddt import DDT
from pyroclast.cpvae.util import build_saveable_objects
from pyroclast.util import direct

from pympler import muppy, summary

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def setup(data_dict,
          optimizer,
          encoder,
          decoder,
          learning_rate,
          latent_dim,
          output_dist,
          max_tree_depth,
          max_tree_leaf_nodes,
          output_dir,
          oversample,
          debug=False,
          expect_load=False):
    num_classes = data_dict['num_classes']
    num_channels = data_dict['shape'][-1]

    # setup model vars
    model_dir = osp.join(output_dir, 'model')
    objects = build_saveable_objects(optimizer_name=optimizer,
                                     encoder_name=encoder,
                                     decoder_name=decoder,
                                     learning_rate=learning_rate,
                                     num_classes=num_classes,
                                     num_channels=num_channels,
                                     latent_dim=latent_dim,
                                     output_dist=output_dist,
                                     max_tree_depth=max_tree_depth,
                                     model_dir=model_dir,
                                     model_name=encoder + decoder)

    model = objects['model']
    optimizer = objects['optimizer']
    global_step = objects['global_step']
    checkpoint = objects['checkpoint']
    ckpt_manager = objects['ckpt_manager']
    writer = tf.summary.create_file_writer(output_dir)

    # load trained model, if available
    if tf.train.latest_checkpoint(model_dir):
        status = checkpoint.restore(tf.train.latest_checkpoint(model_dir))
        print("loaded a model from disk at",
              tf.train.latest_checkpoint(model_dir))
    elif expect_load:
        raise Exception("Model not loaded")

    # train a ddt
    model.classifier.update_model_tree(data_dict['train'],
                                       model.posterior,
                                       oversample=oversample,
                                       debug=debug)
    model.classifier.save_dot(output_dir, 'initial')
    return model, optimizer, global_step, writer, checkpoint, ckpt_manager


def outer_run_minibatch(
    model,
    optimizer,
    global_step,
    alpha,
    beta,
    gamma,
    writer,
    clip_norm=0.,
    is_debug=False,
):

    def run_minibatch(epoch, data, labels, is_train=True, prefix='train'):
        print("Tracing! {} {} {} {}".format(epoch, data, labels, is_train))
        x = tf.cast(data, tf.float32) / 255.
        labels = tf.cast(labels, tf.int32)

        with tf.GradientTape() as tape:
            global_step.assign_add(1)
            x_hat_loc, y_hat, z_posterior, x_hat_scale = model(x)
            y_hat = tf.cast(y_hat, tf.float32)  # from double to single fp

            distortion, rate = model.vae_loss(x,
                                              x_hat_loc,
                                              x_hat_scale,
                                              z_posterior,
                                              y=labels)
            classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=y_hat)
            # classification_loss = classification_loss * float(epoch > gamma_delay)
            loss = tf.reduce_mean(alpha * distortion + beta * rate +
                                  gamma * classification_loss)

        # calculate gradients for current loss
        if is_train:
            gradients = tape.gradient(loss, model.trainable_variables)
            if clip_norm:
                clipped_gradients, _ = tf.clip_by_global_norm(
                    gradients, clip_norm)
            else:
                clipped_gradients = gradients
            optimizer.apply_gradients([
                (grad, var)
                for (grad,
                     var) in zip(clipped_gradients, model.trainable_variables)
                if grad is not None
            ])

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
            tf.summary.scalar(prefix + "loss/mean classification loss",
                              gamma * tf.reduce_mean(classification_loss),
                              step=global_step)
            tf.summary.scalar(prefix + "classification_rate",
                              classification_rate,
                              step=global_step)
            tf.summary.scalar(prefix + "loss/total loss",
                              loss,
                              step=global_step)

        loss_numerator = tf.reduce_sum(alpha * distortion + beta * rate +
                                       gamma * classification_loss)
        classification_rate_numerator = tf.reduce_sum(
            tf.cast(tf.equal(prediction, labels), tf.float32))
        loss_denominator = x.shape[0]
        return loss_numerator, classification_rate_numerator, loss_denominator

    return run_minibatch


def eval(
    data_dict,
    model,
    optimizer,
    global_step,
    writer,
    alpha,
    beta,
    gamma,
    clip_norm,
    tree_update_period,
    num_samples,
    checkpoint,
    ckpt_manager,
    output_dir,
    oversample,
    debug,
):
    run_minibatch_fn = outer_run_minibatch(model,
                                           optimizer,
                                           global_step,
                                           alpha,
                                           beta,
                                           gamma,
                                           writer,
                                           clip_norm,
                                           is_debug=debug)
    run_minibatch_fn = tf.function(run_minibatch_fn)
    # test
    loss_numerator = 0
    classification_rate_numerator = 0
    loss_denominator = 0
    test_batches = data_dict['test']
    if debug:
        test_batches = tqdm(test_batches)
    for batch in test_batches:
        loss_n, class_rate_n, loss_d = run_minibatch_fn(
            epoch=tf.constant(0),
            data=batch['image'],
            labels=batch['label'],
            is_train=tf.constant(False))
        loss_numerator += loss_n
        classification_rate_numerator += class_rate_n
        loss_denominator += loss_d
    return {
        'loss':
            float(loss_numerator) / float(loss_denominator),
        'classification_rate':
            float(classification_rate_numerator) / float(loss_denominator)
    }


def sample(model, num_samples, epoch, output_dir):
    for i in range(num_samples):
        im = np.squeeze(model.sample()[0])
        im = np.minimum(1., np.maximum(0., im))
        im = Image.fromarray((255. * im).astype(np.uint8))
        im.save(
            os.path.join(output_dir, "epoch_{}_sample_{}.png".format(epoch, i)))


def train(data_dict, model, optimizer, global_step, writer, early_stopping,
          alpha, beta, gamma, clip_norm, tree_update_period, num_samples,
          checkpoint, ckpt_manager, output_dir, oversample, debug):
    output_log_file = "file://" + osp.join(output_dir, 'train_log.txt')
    run_minibatch_fn = outer_run_minibatch(model,
                                           optimizer,
                                           global_step,
                                           alpha,
                                           beta,
                                           gamma,
                                           writer,
                                           clip_norm,
                                           is_debug=debug)
    run_minibatch_fn = tf.function(run_minibatch_fn)
    # run training loop
    train_batches = data_dict['train']
    if debug:
        train_batches = tqdm(train_batches, total=data_dict['train_bpe'])
    test_batches = data_dict['test']
    if debug:
        test_batches = tqdm(test_batches, total=data_dict['test_bpe'])
    for epoch in range(early_stopping.max_epochs):
        # train
        tf.print("Epoch", epoch)
        tf.print("Epoch", epoch, output_stream=output_log_file)
        tf.print("TRAIN", output_stream=output_log_file)
        loss_numerator = 0
        loss_denominator = 0
        classification_rate_numerator = 0
        for batch in train_batches:
            loss_n, class_rate_n, loss_d = run_minibatch_fn(
                epoch=tf.constant(epoch),
                data=batch['image'],
                labels=batch['label'],
                is_train=tf.constant(True))
            loss_numerator += loss_n
            classification_rate_numerator += class_rate_n
            loss_denominator += loss_d
        tf.print("loss:",
                 float(loss_numerator) / float(loss_denominator),
                 output_stream=output_log_file)
        tf.print("classification_rate:",
                 float(classification_rate_numerator) / float(loss_denominator),
                 output_stream=output_log_file)

        # test
        loss_numerator = 0
        loss_denominator = 0
        classification_rate_numerator = 0
        tf.print("TEST", output_stream=output_log_file)
        for batch in test_batches:
            loss_n, class_rate_n, loss_d = run_minibatch_fn(
                epoch=tf.constant(epoch),
                data=batch['image'],
                labels=batch['label'],
                is_train=tf.constant(False))
            loss_numerator += loss_n
            loss_denominator += loss_d
            classification_rate_numerator += class_rate_n
        tf.print("loss:",
                 float(loss_numerator) / float(loss_denominator),
                 output_stream=output_log_file)
        tf.print("classification_rate:",
                 float(classification_rate_numerator) / float(loss_denominator),
                 output_stream=output_log_file)

        # sample
        if debug:
            tf.print('Sampling')
        sample(model, num_samples, epoch, output_dir)

        # save parameters
        if early_stopping(epoch,
                          float(loss_numerator) / float(loss_denominator)):
            break

        # update
        if type(model.classifier) is DDT and epoch % tree_update_period == 0:
            if debug:
                tf.print('Updating decision tree')
            score = model.classifier.update_model_tree(data_dict['train'],
                                                       model.posterior,
                                                       oversample=oversample,
                                                       debug=debug)
            tf.print("Accuracy at DDT fit from sampling:",
                     score,
                     output_stream=output_log_file)
            model.classifier.save_dot(output_dir, epoch)

    return model


def learn(
    data_dict,
    encoder,
    decoder,
    seed=None,
    latent_dim=64,
    epochs=1000,
    oversample=1,
    max_tree_depth=5,
    max_tree_leaf_nodes=16,
    tree_update_period=3,
    optimizer='rmsprop',  # adam or rmsprop
    learning_rate=3e-4,
    output_dist='l2',  # disc_logistic or l2 or bernoulli
    output_dir='./',
    num_samples=5,
    clip_norm=0.,
    alpha=1.,
    beta=1.,
    gamma=1.,
    gamma_delay=0,
    patience=12,
    debug=False):
    model, optimizer, global_step, writer, checkpoint, ckpt_manager = setup(
        data_dict,
        optimizer,
        encoder,
        decoder,
        learning_rate,
        latent_dim,
        output_dist,
        max_tree_depth,
        max_tree_leaf_nodes,
        output_dir=output_dir,
        oversample=oversample,
        debug=debug)

    early_stopping = EarlyStopping(patience,
                                   ckpt_manager,
                                   eps=0.03,
                                   max_epochs=epochs)
    model = train(data_dict, model, optimizer, global_step, writer,
                  early_stopping, alpha, beta, gamma, clip_norm,
                  tree_update_period, num_samples, checkpoint, ckpt_manager,
                  output_dir, oversample, debug)
    return model


@direct
def direct_leaf_walk(
    data_dict,
    encoder,
    decoder,
    seed,
    latent_dim,
    epochs,
    oversample,
    max_tree_depth,
    max_tree_leaf_nodes,
    tree_update_period,
    optimizer,  # adam or rmsprop
    learning_rate,
    output_dist,  # disc_logistic or l2 or bernoulli
    output_dir,
    num_samples,
    clip_norm,
    alpha,
    beta,
    gamma,
    gamma_delay=0,
    patience=12,
    debug=False):
    model, optimizer, global_step, writer, checkpoint, ckpt_manager = setup(
        data_dict,
        optimizer,
        encoder,
        decoder,
        learning_rate,
        latent_dim,
        output_dist,
        max_tree_depth,
        max_tree_leaf_nodes,
        output_dir,
        oversample,
        debug,
        expect_load=True)
    loss = eval(data_dict, model, optimizer, global_step, writer, alpha, beta,
                gamma, clip_norm, tree_update_period, num_samples, checkpoint,
                ckpt_manager, output_dir, oversample, debug)


@direct
def direct_eval(
    data_dict,
    encoder,
    decoder,
    seed,
    latent_dim,
    epochs,
    oversample,
    max_tree_depth,
    max_tree_leaf_nodes,
    tree_update_period,
    optimizer,  # adam or rmsprop
    learning_rate,
    output_dist,  # disc_logistic or l2 or bernoulli
    output_dir,
    num_samples,
    clip_norm,
    alpha,
    beta,
    gamma,
    gamma_delay=0,
    patience=12,
    debug=False):
    model, optimizer, global_step, writer, checkpoint, ckpt_manager = setup(
        data_dict,
        optimizer,
        encoder,
        decoder,
        learning_rate,
        latent_dim,
        output_dist,
        max_tree_depth,
        max_tree_leaf_nodes,
        output_dir,
        oversample,
        debug,
        expect_load=True)
    loss = eval(data_dict, model, optimizer, global_step, writer, alpha, beta,
                gamma, clip_norm, tree_update_period, num_samples, checkpoint,
                ckpt_manager, output_dir, oversample, debug)
    import json
    with open(osp.join(output_dir, 'final_loss.json'), 'w') as json_file:
        json.dump(loss, json_file)


@direct
def direct_learn(
    data_dict,
    encoder,
    decoder,
    seed=None,
    latent_dim=64,
    epochs=1000,
    oversample=10,
    max_tree_depth=5,
    max_tree_leaf_nodes=16,
    tree_update_period=3,
    optimizer='rmsprop',  # adam or rmsprop
    learning_rate=3e-4,
    output_dist='l2',  # disc_logistic or l2 or bernoulli
    output_dir='./',
    num_samples=5,
    clip_norm=0.,
    alpha=1.,
    beta=1.,
    gamma=1.,
    gamma_delay=0,
    patience=12,
    batch_size=128,
    debug=False):
    tf.random.set_seed(seed)
    model, optimizer, global_step, writer, checkpoint, ckpt_manager = setup(
        data_dict, optimizer, encoder, decoder, learning_rate, latent_dim,
        output_dist, max_tree_depth, max_tree_leaf_nodes, output_dir,
        oversample, debug)

    early_stopping = EarlyStopping(patience,
                                   ckpt_manager,
                                   eps=0.03,
                                   max_epochs=epochs)
    model = train(data_dict, model, optimizer, global_step, writer,
                  early_stopping, alpha, beta, gamma, clip_norm,
                  tree_update_period, num_samples, checkpoint, ckpt_manager,
                  output_dir, oversample, debug)
    return model
