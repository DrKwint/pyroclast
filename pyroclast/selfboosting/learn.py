import argparse
import functools
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorwatch as tw
from tqdm import tqdm

from pyroclast.selfboosting.graph import (ResidualBoostingModule,
                                          SequentialResNet)
from pyroclast.selfboosting.loss import (
    UpdateMulticlassDistribution, binary_loss, calculate_binary_gamma_tilde,
    calculate_multiclass_gamma_tilde, initial_multiclass_distribution,
    multiclass_loss, update_binary_distribution)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def repr_module(channels):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            channels, 3, padding='same', activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            channels, 3, padding='same', activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization()
    ])


def classification_module(num_classes):
    if num_classes == 1:
        activation = tf.nn.tanh
    else:
        activation = None
    return tf.keras.layers.Dense(
        num_classes,
        kernel_initializer=tf.keras.initializers.RandomNormal(),
        activation=activation)


def prepare_data(dataset, batch_size):
    # load data
    data, info = tfds.load(dataset, with_info=True)
    train_batches_per_epoch = info.splits['train'].num_examples // batch_size
    test_batches_per_epoch = info.splits['test'].num_examples // batch_size
    data_shape = info.features['image'].shape
    dataset_train = data['train']
    dataset_test = data['test']

    dataset_train = dataset_train.shuffle(1024).batch(batch_size).prefetch(
        tf.data.experimental.AUTOTUNE)
    dataset_test = dataset_test.batch(batch_size).prefetch(
        tf.data.experimental.AUTOTUNE)
    return dataset_train, train_batches_per_epoch, dataset_test, test_batches_per_epoch


def learn(dataset,
          seed=None,
          batch_size=32,
          learning_rate=1e-3,
          num_classes=10,
          num_channels=8,
          epochs_per_module=1,
          tb_dir='./tb/'):
    del seed  # currently unused
    if num_classes == 1:
        distribution_update_fn = update_binary_distribution
        loss_fn = binary_loss
        classification_fn = lambda x: tf.cast(
                            tf.math.sign(tf.squeeze(x), 0.5),
                            tf.int64)
        gamma_tilde_calculation_fn = calculate_binary_gamma_tilde
    else:
        initial_distribution_fn = functools.partial(
            initial_multiclass_distribution, num_classes=num_classes)
        distribution_update_fn = UpdateMulticlassDistribution()
        loss_fn = multiclass_loss
        classification_fn = lambda x: tf.argmax(x, axis=1)
        gamma_tilde_calculation_fn = calculate_multiclass_gamma_tilde

    model = SequentialResNet(num_classes, num_channels,
                             initial_distribution_fn, distribution_update_fn,
                             gamma_tilde_calculation_fn)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    dataset_train, train_batches_per_epoch, dataset_test, test_batches_per_epoch = prepare_data(
        dataset, batch_size)

    # tensorboard
    global_step = tf.compat.v1.train.get_or_create_global_step()
    writer = tf.contrib.summary.create_file_writer(tb_dir)
    writer.set_as_default()

    for num_module in range(10):
        # add module
        alpha = tf.ones(10)
        module = ResidualBoostingModule(
            repr_module(num_channels),
            classification_module(num_classes),
            alpha=alpha,
            name='module_{}'.format(num_module))
        model.add_module(module)

        for epoch in range(epochs_per_module):
            print("Module: {} Epoch: {}".format(num_module, epoch))
            print("TRAIN")
            for batch in tqdm(dataset_train, total=train_batches_per_epoch):
                global_step.assign_add(1)
                x = tf.cast(batch['image'], tf.float32) / 255.
                label = batch['label']
                if type(distribution_update_fn) is UpdateMulticlassDistribution:
                    distribution_update_fn.state = 0.
                # train module
                with tf.GradientTape() as tape:
                    boosted_classifiers, weak_module_classifiers, gamma_tildes, gammas, alphas = model(
                        x, label)
                    losses = [
                        tf.reduce_mean(loss_fn(h, label, s)) for (h, s) in zip(
                            weak_module_classifiers, boosted_classifiers)
                    ]
                # calculate gradients for current loss
                gradients = tape.gradient(losses[-1],
                                          module.trainable_variables)
                _, global_norm = tf.clip_by_global_norm(gradients, 10.0)
                optimizer.apply_gradients(
                    zip(gradients, module.trainable_variables))
                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar("gradient_global_norm",
                                              global_norm)
                    [
                        tf.contrib.summary.scalar(
                            "mean_train_loss_module_{}".format(i),
                            loss,
                            family='loss') for (i, loss) in enumerate(losses)
                    ]
                    correct_predictions = [
                        tf.equal(classification_fn(bc), label)
                        for bc in boosted_classifiers
                    ]
                    accuracies = [
                        tf.reduce_mean(tf.cast(cp, tf.float32))
                        for cp in correct_predictions
                    ]
                    [
                        tf.contrib.summary.scalar(
                            "mean_train_boosted_classifier_accuracy_module_{}".
                            format(i),
                            a,
                            family='accuracy')
                        for (i, a) in enumerate(accuracies)
                    ]
                    for (m, a) in enumerate(alphas):
                        a_val = a.numpy()
                        [
                            tf.contrib.summary.scalar(
                                "alpha_module_{}".format(m),
                                val,
                                family='alpha_class_{}'.format(c))
                            for c, val in enumerate(a_val)
                        ]
                    [
                        tf.contrib.summary.scalar(
                            "train_gamma_tilde_module_{}".format(i),
                            a,
                            family='gamma_tilde')
                        for (i, a) in enumerate(gamma_tildes)
                    ]
                    [
                        tf.contrib.summary.scalar(
                            "train_gamma_{}-{}".format(i, i + 1),
                            a,
                            family='gamma') for (i, a) in enumerate(gammas)
                    ]
            print("TEST")
            batch_accuracies = []
            for batch in tqdm(dataset_test, total=test_batches_per_epoch):
                x = tf.cast(batch['image'], tf.float32) / 255.
                label = batch['label']
                if type(distribution_update_fn) is UpdateMulticlassDistribution:
                    distribution_update_fn.state = 0.
                boosted_classifiers, weak_module_classifiers, gamma_tildes, gammas, alphas = model(
                    x, label)
                correct_predictions = [
                    tf.equal(classification_fn(bc), label)
                    for bc in boosted_classifiers
                ]
                accuracies = [
                    tf.reduce_mean(tf.cast(cp, tf.float32))
                    for cp in correct_predictions
                ]
                batch_accuracies.append(accuracies)
            mean_epoch_accuracies = np.mean(
                np.squeeze(np.array(list(zip(batch_accuracies)))), axis=0)
            print("Test accuracy per boosted classifier:",
                  mean_epoch_accuracies)
            with tf.contrib.summary.always_record_summaries():
                [
                    tf.contrib.summary.scalar(
                        "mean_test_boosted_classifier_accuracy_module_{}".
                        format(i),
                        a,
                        family='test/accuracy')
                    for (i, a) in enumerate(mean_epoch_accuracies)
                ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--epochs_per_module', default=1, type=int)
    args = parser.parse_args()
    learn(dataset=args.dataset, epochs_per_module=args.epochs_per_module)
