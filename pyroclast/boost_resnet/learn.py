import functools

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from pyroclast.boost_resnet.graph import (ResidualBoostingModule,
                                          SequentialResNet)
from pyroclast.boost_resnet.loss import (
    UpdateMulticlassDistribution, binary_loss, calculate_binary_gamma_tilde,
    calculate_multiclass_gamma_tilde, initial_multiclass_distribution,
    multiclass_loss, update_binary_distribution)
from pyroclast.boost_resnet.models import repr_module, classification_module


def learn(data_dict,
          seed=None,
          batch_size=32,
          learning_rate=1e-3,
          num_classes=10,
          num_channels=8,
          epochs_per_module=5,
          num_modules=10,
          tb_dir='./tb/'):
    del seed  # currently unused
    num_classes = data_dict['num_classes']

    # binary or multiclass
    if num_classes == 1:
        distribution_update_fn = update_binary_distribution
        loss_fn = binary_loss
        classification_fn = lambda x: tf.cast(tf.math.sign(tf.squeeze(x), 0.5),
                                              tf.int64)
        gamma_tilde_calculation_fn = calculate_binary_gamma_tilde
    else:
        initial_distribution_fn = functools.partial(
            initial_multiclass_distribution, num_classes=num_classes)
        distribution_update_fn = UpdateMulticlassDistribution()
        loss_fn = multiclass_loss
        classification_fn = lambda x: tf.argmax(x, axis=1)
        gamma_tilde_calculation_fn = calculate_multiclass_gamma_tilde

    # setup model
    model = SequentialResNet(num_classes, num_channels, initial_distribution_fn,
                             distribution_update_fn, gamma_tilde_calculation_fn)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # tensorboard
    global_step = tf.compat.v1.train.get_or_create_global_step()
    writer = tf.contrib.summary.create_file_writer(tb_dir)
    writer.set_as_default()

    # training loop
    for num_module in range(num_modules):
        # add module
        alpha = tf.ones(num_classes)
        module = ResidualBoostingModule(
            repr_module(num_channels),
            classification_module(num_classes),
            alpha=alpha,
            name='module_{}'.format(num_module))
        model.add_module(module)

        for epoch in range(epochs_per_module):
            print("Module: {} Epoch: {}".format(num_module, epoch))
            print("TRAIN")
            for batch in tqdm(data_dict['train'], total=data_dict['train_bpe']):
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
            for batch in tqdm(data_dict['test'], total=data_dict['test_bpe']):
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
