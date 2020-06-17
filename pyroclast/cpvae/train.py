import os.path as osp

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import tensorflow_probability as tfp

from pyroclast.cpvae.vqvae import AuxiliaryPrior
from pyroclast.cpvae.util import load_args_from_dir, build_checkpoint
from pyroclast.common.early_stopping import EarlyStopping
from pyroclast.cpvae.util import build_vqvae, build_vae, build_saveable_objects
from pyroclast.cpvae.distribution_classifier import build_ddt_classifier, build_linear_classifier, build_nonlinear_classifier


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
            tf.summary.image(prefix + '/posterior_sample',
                             output_dict['recon'],
                             step=global_step,
                             max_outputs=1)


def outer_run_minibatch(gen_model,
                        class_model,
                        optimizer,
                        global_step,
                        writer,
                        clip_norm=0.):

    def run_eval_minibatch(x, labels):
        global_step.assign_add(1)
        outputs = gen_model.forward_loss(x)
        if class_model is not None:
            outputs.update(class_model.forward_loss(outputs['z'], labels))
        outputs['labels'] = labels
        if hasattr(gen_model, 'output_point_estimate'):
            outputs['recon'] = tf.concat(
                [x + 0.5, gen_model.output_point_estimate(x) + 0.5], -2)
        write_tensorboard(writer, outputs, global_step, prefix='eval')
        loss = outputs['gen_loss']
        if 'class_loss' in outputs:
            loss += outputs['class_loss']
            outputs['total_loss'] = loss
        num_samples = x.shape[0]
        return loss, num_samples

    def run_train_minibatch(x, labels):
        # calculate gradients for current loss
        with tf.GradientTape() as tape:
            global_step.assign_add(1)
            outputs = gen_model.forward_loss(x)
            if class_model is not None:
                outputs.update(class_model.forward_loss(outputs['z'], labels))
            loss = outputs['gen_loss']
            if 'class_loss' in outputs:
                loss += outputs['class_loss']
                outputs['total_loss'] = loss
            gradients = tape.gradient(loss, gen_model.trainable_variables)
        if clip_norm:
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
        else:
            clipped_gradients = gradients
        optimizer.apply(clipped_gradients, gen_model.trainable_variables)

        if hasattr(gen_model, 'output_point_estimate'):
            outputs['recon'] = tf.concat(
                [x + 0.5, gen_model.output_point_estimate(x) + 0.5], -2)
        outputs['labels'] = labels
        write_tensorboard(writer, outputs, global_step, prefix='train')

        num_samples = x.shape[0]
        return loss, num_samples

    return run_train_minibatch, run_eval_minibatch


def train(data_dict, gen_model, class_model, optimizer, global_step,
          ckpt_manager, max_epochs, patience, clip_norm, output_dir, debug):
    output_log_file = "file://" + osp.join(output_dir, 'train_log.txt')
    writer = tf.summary.create_file_writer(output_dir)
    train_minibatch_fn, eval_minibatch_fn = outer_run_minibatch(
        gen_model, class_model, optimizer, global_step, writer, clip_norm)
    train_minibatch_fn = tf.function(train_minibatch_fn)
    eval_minibatch_fn = tf.function(eval_minibatch_fn)
    # run training loop
    train_batches = data_dict['train']
    if debug:
        train_batches = tqdm(train_batches, total=data_dict['train_bpe'])
    test_batches = data_dict['test']
    if debug:
        test_batches = tqdm(test_batches, total=data_dict['test_bpe'])
    early_stopping = EarlyStopping(patience,
                                   ckpt_manager,
                                   max_epochs=max_epochs)
    for epoch in range(early_stopping.max_epochs):

        # train
        tf.print("Epoch", epoch)
        tf.print("Epoch", epoch, output_stream=output_log_file)
        tf.print("TRAIN", output_stream=output_log_file)
        sum_loss = 0
        sum_num_samples = 0
        for batch in train_batches:
            loss, num_samples = train_minibatch_fn(x=batch['image'],
                                                   labels=batch['label'])
            sum_loss += loss
            sum_num_samples += num_samples
        tf.print("loss:",
                 float(sum_loss) / float(sum_num_samples),
                 output_stream=output_log_file)

        # test
        sum_loss = 0
        sum_num_samples = 0
        tf.print("TEST", output_stream=output_log_file)
        for batch in test_batches:
            loss, num_samples = eval_minibatch_fn(x=batch['image'],
                                                  labels=batch['label'])
            sum_loss += loss
            sum_num_samples += num_samples
        tf.print("loss:",
                 float(sum_loss) / float(sum_num_samples),
                 output_stream=output_log_file)

        # sample
        #if debug:
        #    tf.print('Sampling')
        #sample(model, num_samples, epoch, output_dir)

        # save parameters and do early stopping
        print(float(sum_loss) / float(sum_num_samples))
        if early_stopping(epoch, float(sum_loss) / float(sum_num_samples)):
            break

        # update
        """
        if isinstance(model.classifier,
                      DDT) and epoch % tree_update_period == 0:
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
            joblib.dump(model.classifier, osp.join(output_dir, 'ddt.joblib'))
            model.prior = model.classifier.tree_distribution
        """

    return gen_model


def learn_vae(data_dict,
              seed,
              encoder,
              decoder,
              prior,
              posterior,
              output_distribution,
              latent_dim,
              optimizer,
              batch_size,
              max_epochs,
              patience,
              learning_rate,
              model_name,
              output_dir,
              save_dir,
              beta=1.,
              class_loss_coeff=1.,
              debug=False):
    for x in data_dict['train']:
        data_channels = x['image'].shape[-1]
        num_classes = 10
        break

    gen_model = build_vae(encoder, decoder, prior, posterior,
                          output_distribution, latent_dim, data_channels, beta)
    class_model = build_linear_classifier(num_classes,
                                          class_loss_coeff=class_loss_coeff)
    objects = build_saveable_objects(optimizer, learning_rate, model_name,
                                     gen_model, class_model, save_dir)
    global_step = objects['global_step']
    ckpt_manager = objects['ckpt_manager']
    optimizer = objects['optimizer']
    return train(data_dict, gen_model, class_model, optimizer, global_step,
                 ckpt_manager, max_epochs, patience, None, output_dir, debug)


def learn_vqvae(data_dict,
                seed,
                encoder,
                decoder,
                layers,
                optimizer,
                batch_size,
                embedding_dim,
                num_embeddings,
                commitment_cost,
                max_epochs,
                patience,
                learning_rate,
                output_dir,
                save_dir,
                class_loss_coeff,
                load_dir=None,
                debug=False):
    objects = setup_vqvae(data_dict, encoder, decoder, layers, optimizer,
                          embedding_dim, num_embeddings, commitment_cost,
                          learning_rate, save_dir, class_loss_coeff, load_dir)
    gen_model = objects['gen_model']
    class_model = objects['class_model']
    optimizer = objects['optimizer']
    global_step = objects['global_step']
    ckpt_manager = objects['ckpt_manager']

    return train(data_dict, gen_model, class_model, optimizer, global_step,
                 ckpt_manager, max_epochs, patience, None, output_dir, debug)


def learn_vqvae_prior(data_dict,
                      seed,
                      param_dir,
                      save_dir,
                      load_model_dir,
                      output_dir,
                      max_epochs,
                      patience,
                      debug=False,
                      **kwargs):
    args = load_args_from_dir(param_dir)
    objects = setup_vqvae(data_dict, args['encoder'], args['decoder'],
                          args['layers'], args['optimizer'],
                          args['embedding_dim'], args['num_embeddings'],
                          args['commitment_cost'], args['learning_rate'],
                          save_dir, args['class_loss_coeff'], load_model_dir)

    optimizer = objects['optimizer']
    global_step = objects['global_step']
    gen_model = objects['gen_model']
    ckpt_manager = objects['ckpt_manager']

    # TODO update data dict with a map using the vqvae
    def bottom_code(batch):
        outs = gen_model(batch['image'])
        code = tf.expand_dims(outs['vq_output_bottom']['encoding_indices'], -1)
        code = tf.cast(code, tf.float32)
        if code.shape[1] % 2 == 1:
            code = tf.pad(code, [[0, 0], [0, 1], [0, 1], [0, 0]])
        batch['image'] = code
        return batch

    def top_code(batch):
        outs = gen_model(batch['image'])
        code = tf.expand_dims(outs['vq_output_bottom']['encoding_indices'], -1)
        code = tf.cast(code, tf.float32)
        if code.shape[1] % 2 == 1:
            code = tf.pad(code, [[0, 0], [0, 1], [0, 1], [0, 0]])
        batch['image'] = code
        return batch

    # this does nothing except to initialize the model variables
    for d in data_dict['train']:
        gen_model(d['image'])
        break

    if gen_model.num_layers == 2:
        code_fn = top_code
        num_embeddings = gen_model._vq_top.num_embeddings
    else:
        code_fn = bottom_code
        num_embeddings = gen_model._vq_bottom.num_embeddings
    data_dict['train'] = data_dict['train'].map(code_fn)
    data_dict['test'] = data_dict['test'].map(code_fn)

    for d in data_dict['train']:
        code_shape = d['image'].shape[1:]
        break

    aux_prior = AuxiliaryPrior(code_shape, num_embeddings)
    train(data_dict, aux_prior, None, optimizer, global_step, ckpt_manager,
          max_epochs, patience, None, output_dir, debug)


def setup_vqvae(data_dict,
                encoder,
                decoder,
                layers,
                optimizer,
                embedding_dim,
                num_embeddings,
                commitment_cost,
                learning_rate,
                save_dir,
                class_loss_coeff,
                load_model_dir=None):
    for d in data_dict['train']:
        output_channels = d['image'].shape[-1]
        break
    num_classes = 10
    train_images = np.array(
        [d['image'] for d in data_dict['train'].unbatch().as_numpy_iterator()])
    train_data_variance = np.var(train_images)

    objects = dict()
    objects['gen_model'] = build_vqvae(encoder,
                                       decoder,
                                       layers,
                                       output_channels,
                                       train_data_variance,
                                       embedding_dim,
                                       num_embeddings,
                                       commitment_cost,
                                       output_channels=output_channels)
    if class_loss_coeff > 0.:
        objects['class_model'] = build_linear_classifier(
            num_classes, class_loss_coeff)

    # load model
    if load_model_dir is not None:
        checkpoint, ckpt_manager = build_checkpoint(objects, load_model_dir)
        status = checkpoint.restore(ckpt_manager.latest_checkpoint)

    objects.update(
        build_saveable_objects(optimizer, learning_rate, objects, save_dir))

    return objects
