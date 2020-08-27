import os.path as osp

import sonnet as snt
import tensorflow as tf

from ..common.early_stopping import EarlyStopping
from ..common.tf_util import build_checkpoint
from .mypixelcnn import PixelSNAIL
from .util import load_args_from_dir, train
from .vq_decoder import HalfDecoder
from .vq_encoder import HalfEncoder
from .vqvae import VQVAE


def write_tensorboard(writer, output_dict, global_step, prefix='train'):
    with writer.as_default():
        if 'y_hat' in output_dict and 'labels' in output_dict:
            prediction = tf.math.argmax(output_dict['y_hat'],
                                        axis=1,
                                        output_type=tf.int32)
            classification_rate = tf.reduce_mean(
                tf.cast(tf.equal(prediction, output_dict['labels']),
                        tf.float32))
            tf.summary.scalar(prefix + "/classification_rate",
                              classification_rate,
                              step=global_step)
        if 'class_loss' in output_dict:
            tf.summary.scalar(prefix + "/mean classification loss",
                              tf.reduce_mean(output_dict['class_loss']),
                              step=global_step)
        if 'recon_loss' in output_dict:
            tf.summary.scalar(prefix + "/mean reconstruction loss",
                              tf.reduce_mean(output_dict['recon_loss']),
                              step=global_step)
        if 'latent_loss' in output_dict:
            tf.summary.scalar(prefix + "/mean latent loss",
                              tf.reduce_mean(output_dict['latent_loss']),
                              step=global_step)

        if 'vqs' in output_dict:
            for i, vq in enumerate(output_dict['vqs']):
                tf.summary.scalar(prefix +
                                  "/mean codebook {} perplexity".format(i),
                                  tf.reduce_mean(vq['perplexity']),
                                  step=global_step)
                tf.summary.scalar(prefix + "/mean codebook {} loss".format(i),
                                  tf.reduce_mean(vq['loss']),
                                  step=global_step)
        if 'vq_output_top' in output_dict:
            tf.summary.scalar(prefix + "/mean top codebook perplexity",
                              tf.reduce_mean(
                                  output_dict['vq_output_top']['perplexity']),
                              step=global_step)
            tf.summary.scalar(prefix + "/mean top codebook loss",
                              tf.reduce_mean(
                                  output_dict['vq_output_top']['loss']),
                              step=global_step)
        if 'vq_output_bottom' in output_dict:
            tf.summary.scalar(
                prefix + "/mean bottom codebook perplexity",
                tf.reduce_mean(output_dict['vq_output_bottom']['perplexity']),
                step=global_step)
            tf.summary.scalar(prefix + "/mean bottom codebook loss",
                              tf.reduce_mean(
                                  output_dict['vq_output_bottom']['loss']),
                              step=global_step)
        if 'gen_loss' in output_dict:
            tf.summary.scalar(prefix + "/mean generative loss",
                              tf.reduce_mean(output_dict['gen_loss']),
                              step=global_step)
        if 'total_loss' in output_dict:
            tf.summary.scalar(prefix + "/mean total loss",
                              tf.reduce_mean(output_dict['total_loss']),
                              step=global_step)
        if 'recon' in output_dict:
            if 'data' in output_dict:
                img = tf.concat([output_dict['data'], output_dict['recon']], -2)
            else:
                img = output_dict['recon']
            tf.summary.image(prefix + '/posterior_sample',
                             img,
                             step=global_step)


def outer_run_minibatch(model, optimizer, global_step, writer):

    def run_eval_minibatch(x, labels):
        global_step.assign_add(1)
        outputs = model.forward_loss(x)
        outputs['data'] = x
        write_tensorboard(writer, outputs, global_step, prefix='eval')
        num_samples = x.shape[0]
        return outputs['loss'], num_samples

    def run_train_minibatch(x, labels):
        # calculate gradients for current loss
        with tf.GradientTape() as tape:
            global_step.assign_add(1)
            outputs = model.forward_loss(x)
            loss = outputs['loss']
            gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply(gradients, model.trainable_variables)

        outputs['data'] = x
        write_tensorboard(writer, outputs, global_step, prefix='train')
        num_samples = x.shape[0]
        return loss, num_samples

    return run_train_minibatch, run_eval_minibatch


def load_vqvae(args, data_channels):
    channels = args['channels']
    num_embeddings = args['num_embeddings']
    vq_commitment_cost = args['vq_commitment_cost']
    model = VQVAE(HalfEncoder(channels, num_embeddings, vq_commitment_cost),
                  HalfDecoder(channels, data_channels))
    global_step = tf.compat.v1.train.get_or_create_global_step()
    load_dir = args['output_dir']
    load_objects = {'model': model, 'global_step': global_step}
    checkpoint, _ = build_checkpoint(load_objects, load_dir)
    status = checkpoint.restore(tf.train.latest_checkpoint(load_dir))
    status.assert_existing_objects_matched()
    return model, global_step


def learn_prior(data_dict,
                seed,
                batch_size,
                learning_rate,
                num_embeddings,
                model_args_dir,
                output_dir,
                patience=20,
                max_epochs=1000,
                debug=False):
    args = load_args_from_dir(model_args_dir)
    vqvae, global_step = load_vqvae(args, data_dict['shape'][-1])

    def code_fn(batch, idx=0):
        outs = vqvae(batch['image'])
        code = outs['vqs'][idx]['encoding_indices']
        batch['image'] = code
        return batch

    # this does nothing except to initialize the model variables
    for d in data_dict['train']:
        vqvae(d['image'])
        code_shape = code_fn(d)['image'].shape[1:] + (1,)
        break

    data_dict['train'] = data_dict['train'].map(code_fn)
    data_dict['test'] = data_dict['test'].map(code_fn)

    model = PixelSNAIL(shape=code_shape,
                       n_class=num_embeddings,
                       nr_filters=32,
                       kernel_size=3,
                       n_block=2,
                       n_res_block=2,
                       res_channel=64)
    optimizer = snt.optimizers.Adam(learning_rate=learning_rate)
    writer = tf.summary.create_file_writer(output_dir)
    train_minibatch_fn, test_minibatch_fn = outer_run_minibatch(
        model, optimizer, global_step, writer)
    #train_minibatch_fn = tf.function(train_minibatch_fn)
    #eval_minibatch_fn = tf.function(test_minibatch_fn)
    eval_minibatch_fn = test_minibatch_fn

    save_objects = {'model': model, 'global_step': global_step}
    checkpoint, ckpt_manager = build_checkpoint(save_objects, output_dir)
    early_stopping = EarlyStopping(patience, ckpt_manager, max_epochs)
    output_log_file = "file://" + osp.join(output_dir, 'train_log.txt')
    train(data_dict, train_minibatch_fn, eval_minibatch_fn, early_stopping,
          output_log_file, debug)


def learn(data_dict,
          seed,
          batch_size,
          channels,
          num_embeddings,
          vq_commitment_cost,
          learning_rate,
          output_dir,
          patience,
          max_epochs=1000,
          debug=False):
    model = VQVAE(HalfEncoder(channels, num_embeddings, vq_commitment_cost),
                  HalfDecoder(channels, data_dict['shape'][-1]))

    optimizer = snt.optimizers.Adam(learning_rate=learning_rate)
    writer = tf.summary.create_file_writer(output_dir)
    global_step = tf.compat.v1.train.get_or_create_global_step()
    train_minibatch_fn, test_minibatch_fn = outer_run_minibatch(
        model, optimizer, global_step, writer)
    train_minibatch_fn = tf.function(train_minibatch_fn)
    eval_minibatch_fn = tf.function(test_minibatch_fn)

    save_objects = {'model': model, 'global_step': global_step}
    checkpoint, ckpt_manager = build_checkpoint(save_objects, output_dir)
    early_stopping = EarlyStopping(patience, ckpt_manager, max_epochs)
    output_log_file = "file://" + osp.join(output_dir, 'train_log.txt')
    train(data_dict, train_minibatch_fn, eval_minibatch_fn, early_stopping,
          output_log_file, debug)
