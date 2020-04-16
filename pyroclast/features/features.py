import copy
import os
import re

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




def train(data_dict, model, optimizer, global_step, writer, early_stopping,
          train_conv_stack, lambd, alpha, checkpoint, ckpt_manager, debug):
    if train_conv_stack:
        train_model = model
    else:
        train_model = model.classifier

    base_epoch = global_step // (data_dict['train_bpe'] + data_dict['test_bpe'])
    for epoch in range(base_epoch + 1, early_stopping.max_epochs):
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
          model_name='generic_classifier'):
    objects = build_savable_objects(conv_stack_name, data_dict, learning_rate,
                                    output_dir, model_name)
    model = objects['model']
    optimizer = objects['optimizer']
    global_step = objects['global_step']
    checkpoint = objects['checkpoint']
    ckpt_manager = objects['ckpt_manager']
    early_stopping = EarlyStopping(patience,
                                   ckpt_manager=ckpt_manager,
                                   eps=0.03,
                                   max_epochs=max_epochs)
    if ckpt_manager.latest_checkpoint is not None:
        checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()

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

    train(train_data, model, optimizer, global_step, writer, early_stopping,
          (not is_preprocessed), lambd, alpha, checkpoint, ckpt_manager, debug)

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
    for lambd in [0., 1e0, 1e1, 1e2]:
        for alpha in [0., 1e0, 1e1]:
            if lambd == 0. and alpha > 0.:
                continue
            args = copy.copy(args_template)
            args['lambd'] = lambd
            args['alpha'] = alpha
            model_name = 'mnist_lambd{:1.0e}_alpha{:1.0e}'.format(lambd, alpha)
            objects = build_savable_objects(
                args['conv_stack_name'], args['data_dict'],
                args['learning_rate'],
                os.path.join(args['output_dir'], model_name), model_name)
            model = objects['model']
            checkpoint = objects['checkpoint']
            ckpt_manager = objects['ckpt_manager']
            if ckpt_manager.latest_checkpoint is not None:
                checkpoint.restore(
                    ckpt_manager.latest_checkpoint).expect_partial()
            else:
                print("Wrong directory for output_dir {}?".format(model_name))
                continue
            model_name = 'lambda {}\nalpha {}'.format(lambd, alpha)
            models[model_name] = model

    for batch in data_dict['test']:
        fig = plot_grads(tf.cast(batch['image'][:5], tf.float32) / 255.,
                         list(models.values()),
                         list(models.keys()),
                         data_dict['shape'],
                         data_dict['num_classes'],
                         debug=debug)
        break
    plt.savefig('input_grads.png')


def visualize_feature_perturbations(data_dict,
                                    seed,
                                    output_dir,
                                    debug,
                                    conv_stack_name='tiny_net'):
    objects = build_savable_objects(conv_stack_name, data_dict, 2e-4,
                                    output_dir, 'features_model')
    model = objects['model']
    checkpoint = objects['checkpoint']
    ckpt_manager = objects['ckpt_manager']
    if ckpt_manager.latest_checkpoint is not None:
        checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
    else:
        print("Wrong directory for output_dir {}?".format(output_dir))

    # calculate usefulness
    usefulness = model.usefulness(data_dict['train'].map(
        lambda x: (tf.cast(x['image'], tf.float32), x['label'])),
                                  data_dict['num_classes'],
                                  debug=debug)
    heatmap(usefulness,
            output_dir + '/' + 'mnist_lambd1' + '_rho_usefulness.png',
            'rho usefulness')

    for batch in data_dict['train']:
        if batch['label'][0] != 9:
            continue
        original = tf.cast(tf.expand_dims(batch['image'][0], 0), tf.float32)
        features = model.features(original)
        found = model.input_search(
            original,
            model.features(original) *
            (1. - tf.one_hot(85, features.shape[-1])))
        print('Original value:', tf.squeeze(model.features(original))[85])
        print('Found value:', tf.squeeze(model.features(found))[85])
        print('Original class logits', model.logits(original))
        print('Found class logits', model.logits(found))
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
