import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from pyroclast.common.early_stopping import EarlyStopping
from pyroclast.common.models import get_network_builder
from pyroclast.common.plot import plot_grads
from pyroclast.common.preprocessed_dataset import PreprocessedDataset
from pyroclast.common.util import dummy_context_mgr, heatmap
from pyroclast.features.generic_classifier import GenericClassifier
from pyroclast.features.networks import ross_net


# define minibatch fn
@tf.function
def run_minibatch(model,
                  optimizer,
                  global_step,
                  epoch,
                  batch,
                  num_classes,
                  lambd,
                  alpha,
                  writer,
                  is_train=True):
    """
    Args:
        model (tf.Module):
        optimizer (tf.Optimizer):
        global_step (Tensor):
        epoch (int): Epoch of training for logging
        batch (dict): dict from dataset
        writer (tf.summary.SummaryWriter):
        is_train (bool): Optional, run backwards pass if True
    """
    x = tf.cast(batch['image'], tf.float32) / 255.
    labels = tf.cast(batch['label'], tf.int32)
    with tf.GradientTape() as tape:
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(x)
            y_hat = model(x)

            # classification loss
            classification_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(labels, num_classes), logits=y_hat)

        if lambd != 0.:
            # input gradient regularization
            grad = inner_tape.gradient(y_hat, x)
            input_grad_reg_loss = tf.math.square(tf.norm(grad, 2))

            if alpha != 0.:
                grad_masked_y_hat = model(x * grad)
                grad_masked_classification_loss = tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.one_hot(labels, num_classes),
                    logits=grad_masked_y_hat)
            else:
                grad_masked_classification_loss = 0.
        else:
            input_grad_reg_loss = 0.
            grad_masked_classification_loss = 0.

        # total loss
        total_loss = classification_loss + (lambd * input_grad_reg_loss) + (
            alpha * grad_masked_classification_loss)
        mean_total_loss = tf.reduce_mean(total_loss)
        global_step.assign_add(1)

    if is_train:
        train_vars = model.trainable_variables
        gradients = tape.gradient(mean_total_loss, train_vars)
        optimizer.apply_gradients(zip(gradients, train_vars))

    # log to TensorBoard
    prefix = 'train_' if is_train else 'validate_'
    with writer.as_default():
        prediction = tf.math.argmax(y_hat, axis=1, output_type=tf.int32)
        classification_rate = tf.reduce_mean(
            tf.cast(tf.equal(prediction, labels), tf.float32))
        tf.summary.scalar(prefix + "classification_rate",
                          classification_rate,
                          step=global_step)
        tf.summary.scalar(prefix + "loss/mean classification",
                          tf.reduce_mean(classification_loss),
                          step=global_step)
        tf.summary.scalar(prefix + "loss/mean input gradient regularization",
                          lambd * tf.reduce_mean(input_grad_reg_loss),
                          step=global_step)
        tf.summary.scalar(prefix + "loss/mean total loss",
                          mean_total_loss,
                          step=global_step)
    loss_numerator = tf.reduce_sum(classification_loss)
    accuracy_numerator = tf.reduce_sum(
        tf.cast(tf.equal(prediction, labels), tf.int32))
    denominator = x.shape[0]
    return loss_numerator, accuracy_numerator, denominator


def train(data_dict, model, optimizer, global_step, writer, early_stopping,
          train_conv_stack, lambd, alpha, checkpoint, ckpt_manager, debug):
    if train_conv_stack:
        train_model = model
    else:
        train_model = model.classifier

    # use the max epoch value in early_stopping
    for epoch in range(10000):
        # train
        train_batches = data_dict['train']
        num_classes = data_dict['num_classes']
        if debug:
            train_batches = tqdm(train_batches, total=data_dict['train_bpe'])
        print("Epoch", epoch)
        print("TRAIN")
        loss_numerator = 0
        acc_numerator = 0
        denominator = 0
        for batch in train_batches:
            l, a, d = run_minibatch(train_model,
                                    optimizer,
                                    global_step,
                                    epoch,
                                    batch,
                                    num_classes,
                                    lambd,
                                    alpha,
                                    writer,
                                    is_train=True)
            acc_numerator += a
            loss_numerator += l
            denominator += d
        print("Train Accuracy:", float(acc_numerator) / float(denominator))
        print("Train Loss:", float(loss_numerator) / float(denominator))

        # test
        test_batches = data_dict['test']
        if debug:
            test_batches = tqdm(test_batches, total=data_dict['test_bpe'])
        print("TEST")
        loss_numerator = 0
        acc_numerator = 0
        denominator = 0
        for batch in test_batches:
            l, a, d = run_minibatch(train_model,
                                    optimizer,
                                    global_step,
                                    epoch,
                                    batch,
                                    num_classes,
                                    lambd,
                                    alpha,
                                    writer,
                                    is_train=False)
            acc_numerator += a
            loss_numerator += l
            denominator += d
        print("Test Accuracy:", float(acc_numerator) / float(denominator))
        print("Test Loss:", float(loss_numerator) / float(denominator))

        # checkpointing and early stopping
        if early_stopping(epoch, float(loss_numerator) / float(denominator)):
            break

    # restore best parameters
    checkpoint.restore(ckpt_manager.latest_checkpoint).assert_consumed()


def build_savable_objects(conv_stack_name, data_dict, learning_rate, output_dir,
                          model_name):
    global_step = tf.compat.v1.train.get_or_create_global_step()
    conv_stack = get_network_builder(conv_stack_name)()
    classifier = tf.keras.Sequential(
        [tf.keras.layers.Dense(data_dict['num_classes'])])
    model = GenericClassifier(conv_stack, classifier, model_name)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         beta_1=0.5,
                                         epsilon=10e-4)
    save_dict = {
        model_name + '_optimizer': optimizer,
        model_name + '_model': model,
        model_name + '_global_step': global_step
    }
    checkpoint = tf.train.Checkpoint(**save_dict)

    print('my_dir', os.path.join(output_dir, model_name))
    ckpt_manager = tf.train.CheckpointManager(checkpoint,
                                              directory=os.path.join(
                                                  output_dir, model_name),
                                              max_to_keep=3)
    return {
        'model': model,
        'optimizer': optimizer,
        'global_step': global_step,
        'checkpoint': checkpoint,
        'ckpt_manager': ckpt_manager
    }


def learn(data_dict,
          seed,
          output_dir,
          debug,
          learning_rate=3e-3,
          conv_stack_name='vgg19',
          is_preprocessed=False,
          train_conv_stack=False,
          patience=2,
          max_epochs=10,
          lambd=0.,
          alpha=0.,
          model_save_name='generic_classifier'):
    objects = build_savable_objects(conv_stack_name, data_dict, learning_rate,
                                    output_dir, model_save_name)
    model = objects['model']
    optimizer = objects['optimizer']
    global_step = objects['global_step']
    checkpoint = objects['checkpoint']
    ckpt_manager = objects['ckpt_manager']
    writer = tf.summary.create_file_writer(output_dir)
    # setup checkpointing
    if is_preprocessed:
        for x in data_dict['train']:
            batch_size = x['label'].shape[0]
            break
        preprocessed_dataset = copy.copy(data_dict)
        preprocessed_dataset['train'] = PreprocessedDataset(
            data_dict['train'], model.features, '.preprocessed_data/vgg19' +
            data_dict['name'] + '_train')(batch_size)
        preprocessed_dataset['test'] = PreprocessedDataset(
            data_dict['test'], model.features, '.preprocessed_data/vgg19' +
            data_dict['name'] + '_test')(batch_size)
        train_data = preprocessed_dataset
    else:
        train_data = data_dict

    early_stopping = EarlyStopping(patience,
                                   ckpt_manager,
                                   eps=0.03,
                                   max_epochs=max_epochs)
    train(train_data, model, optimizer, global_step, writer, early_stopping,
          (not is_preprocessed), lambd, alpha, checkpoint, ckpt_manager, debug)

    usefulness = model.usefulness(train_data['test'].map(
        lambda x: (tf.cast(x['image'], tf.float32), x['label'])),
                                  train_data['num_classes'],
                                  is_preprocessed=is_preprocessed)
    heatmap(usefulness, output_dir + '/' + model_name + '_rho_usefulness.png',
            'rho usefulness')

    return model


def plot_input_grads(data_dict,
                     seed,
                     output_dir,
                     debug,
                     conv_stack_name='vgg19'):
    args = locals()
    args['learning_rate'] = 2e-4
    args['train_conv_stack'] = True
    args['patience'] = 5
    args['max_epochs'] = 50

    args_template = args

    models = {}
    if not os.path.exists('mnist_normal'):
        model_name = 'mnist_normal'
        args['output_dir'] = model_name
        models[model_name] = learn(**args)

    for lambd in [1e0, 1e1, 1e2]:
        for alpha in [0., 1e0, 1e1, 1e2]:
            args = copy.copy(args_template)
            args['lambd'] = lambd
            args['alpha'] = alpha
            model_name = 'mnist_lambd{:1.0e}_alpha{:1.0e}'.format(lambd, alpha)
            args['output_dir'] = model_name
            if not os.path.exists(model_name):
                models[model_name] = learn(**args)
            else:
                model, _, _, checkpoint, ckpt_manager = build_savable_objects(
                    args['conv_stack_name'], args['data_dict'],
                    args['learning_rate'], args['output_dir'], model_name)
                if ckpt_manager.latest_checkpoint is not None:
                    checkpoint.restore(ckpt_manager.latest_checkpoint)
                else:
                    print(
                        "Wrong directory for output_dir {}?".format(output_dir))
                models[model_name] = model

    fig = plot_grads(data_dict['test'], models.values(), models.keys(),
                     data_dict['shape'], data_dict['num_classes'])
    plt.savefig('input_grads.png')


def visualize_feature_perturbations(data_dict,
                                    seed,
                                    output_dir,
                                    debug,
                                    conv_stack_name='tiny_net'):
    model, _, _, checkpoint, ckpt_manager = build_savable_objects(
        conv_stack_name, data_dict, 1e-4, output_dir, 'mnist_lambd1')
    if ckpt_manager.latest_checkpoint is not None:
        checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
    else:
        print("Wrong directory for output_dir {}?".format(output_dir))

    # calculate usefulness
    """
    usefulness = model.usefulness(
        data_dict['train'].map(lambda x:
                               (tf.cast(x['image'], tf.float32), x['label'])),
        data_dict['num_classes'])
    heatmap(usefulness,
            output_dir + '/' + 'mnist_lambd1' + '_rho_usefulness.png',
            'rho usefulness')
    """

    for batch in data_dict['train']:
        original = tf.cast(tf.expand_dims(batch['image'][0], 0), tf.float32)
        features_len = model.features(original).shape[0]
        found = model.input_search(
            original,
            model.features(original) + 1. * tf.one_hot(247, features_len))
        break

    plt.imshow(tf.squeeze(original))
    print(tf.reduce_min(original), tf.reduce_max(original))
    plt.savefig('original')
    plt.close()
    plt.imshow(tf.squeeze(found))
    print(tf.reduce_min(found), tf.reduce_max(found))
    plt.savefig('found')
    plt.close()
    plt.imshow(tf.squeeze(found - original))
    print(tf.reduce_min(found - original), tf.reduce_max(found - original))
    plt.savefig('diff')
    plt.close()
