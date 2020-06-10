import os.path as osp

import tensorflow as tf
from tqdm import tqdm
import numpy as np

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
        if 'vq_output' in output_dict:
            tf.summary.scalar(prefix + "/mean codebook perplexity",
                              tf.reduce_mean(
                                  output_dict['vq_output']['perplexity']),
                              step=global_step)
            tf.summary.scalar(prefix + "/mean codebook loss",
                              tf.reduce_mean(output_dict['vq_output']['loss']),
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

    def run_eval_minibatch(data, labels):
        x = tf.cast(data, tf.float32) / 255.
        labels = tf.cast(labels, tf.int32)
        global_step.assign_add(1)
        outputs = gen_model.forward_loss(x)
        outputs.update(class_model.forward_loss(outputs['z'], labels))
        outputs['labels'] = labels
        outputs['recon'] = tf.concat([x, gen_model.output_point_estimate(x)],
                                     -2)
        write_tensorboard(writer, outputs, global_step, prefix='eval')
        loss = outputs['gen_loss']
        if 'class_loss' in outputs:
            loss += outputs['class_loss']
            outputs['total_loss'] = loss
        num_samples = x.shape[0]
        return loss, num_samples

    def run_train_minibatch(data, labels):
        x = tf.cast(data, tf.float32) / 255.
        labels = tf.cast(labels, tf.int32)

        # calculate gradients for current loss
        with tf.GradientTape() as tape:
            global_step.assign_add(1)
            outputs = gen_model.forward_loss(x)
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
        optimizer.apply_gradients([
            (grad, var)
            for (grad,
                 var) in zip(clipped_gradients, gen_model.trainable_variables)
            if grad is not None
        ])

        outputs['recon'] = tf.concat([x, gen_model.output_point_estimate(x)],
                                     -2)
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
            loss, num_samples = train_minibatch_fn(data=batch['image'],
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
            loss, num_samples = eval_minibatch_fn(data=batch['image'],
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
                optimizer,
                batch_size,
                max_epochs,
                patience,
                learning_rate,
                model_name,
                output_dir,
                save_dir,
                class_loss_coeff=1.,
                debug=False):
    # This value is not that important, usually 64 works.
    # This will not change the capacity in the information-bottleneck.
    embedding_dim = 64

    # The higher this value, the higher the capacity in the information bottleneck.
    num_embeddings = 512

    # commitment_cost should be set appropriately. It's often useful to try a couple
    # of values. It mostly depends on the scale of the reconstruction cost
    # (log p(x|z)). So if the reconstruction cost is 100x higher, the
    # commitment_cost should also be multiplied with the same amount.
    commitment_cost = 2.5  #.25

    num_classes = 10
    train_images = np.array(
        [d['image'] for d in data_dict['train'].unbatch().as_numpy_iterator()])
    train_data_variance = np.var(train_images / 255.0)

    gen_model = build_vqvae(encoder,
                            decoder,
                            train_data_variance,
                            embedding_dim,
                            num_embeddings,
                            commitment_cost,
                            layers=2)
    class_model = build_linear_classifier(num_classes, class_loss_coeff)
    objects = build_saveable_objects(optimizer, learning_rate, model_name,
                                     gen_model, class_model, save_dir)
    global_step = objects['global_step']
    ckpt_manager = objects['ckpt_manager']
    optimizer = objects['optimizer']

    return train(data_dict, gen_model, class_model, optimizer, global_step,
                 ckpt_manager, max_epochs, patience, None, output_dir, debug)
