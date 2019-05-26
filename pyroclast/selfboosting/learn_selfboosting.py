import sonnet as snt
import tensorflow as tf
import numpy as np
import functools

from pyroclast import selfboosting
from pyroclast.selfboosting.sequential_resnet import SequentialResNet
from pyroclast.selfboosting.residual_boosting_module import ResidualBoostingModule
from pyroclast.common.tf_util import run_epoch_ops, calculate_accuracy


def train(session,
          train_data_iterator,
          train_batches_per_epoch,
          test_data_iterator,
          test_batches_per_epoch,
          epochs,
          train_op,
          train_verbose_dict,
          test_verbose_dict,
          verbose=True):
    for epoch in range(epochs):
        # re-initialize data iterators
        session.run(train_data_iterator.initializer)
        session.run(test_data_iterator.initializer)

        # run a training epoch
        print("Epoch", epoch)
        train_vals_dict = run_epoch_ops(session,
                                        train_batches_per_epoch,
                                        train_verbose_dict, [train_op],
                                        verbose=verbose)
        print({'mean ' + k: np.mean(v) for k, v in train_vals_dict.items()})

        # run a test epoch
        test_vals_dict = run_epoch_ops(session,
                                       test_batches_per_epoch,
                                       test_verbose_dict,
                                       verbose=verbose)
        print({'mean ' + k: np.mean(v) for k, v in test_vals_dict.items()})


def learn(train_data_iterator,
          train_batches_per_epoch,
          test_data_iterator,
          test_batches_per_epoch,
          seed,
          repr_module_name='conv_block',
          hypothesis_module_name='linear_classifier',
          num_classes=10,
          num_channels=32,
          module_num=10,
          epochs_per_module=3):
    """
    Args:
        train_data_iterator (tf.data.Iterator): initializable iterator over
            (img, label) where img values are in range [0,255]
        train_batches_per_epoch (int): number of iterator calls to get all
            data
        test_data_iterator (tf.data.Iterator): initializable iterator over
            (img, label) where img values are in range [0,255]
        test_batches_per_epoch (int): number of iterator calls to get all
            data
        seed (int): seed for RNG
        epochs (int): max number of epochs to run training
        class_num (int): number of classes in the classification problem
        module_num (int): max number of weak learner modules in boosted
            classifier
        module_name (str): name of module to use as residual block
        representation_channels (int): number of channels to use in residual
            representation

    """

    # initialize iterators and get input tensors
    session = tf.Session()
    session.run(train_data_iterator.initializer)
    train_data = train_data_iterator.get_next()
    train_data['image'] = tf.cast(train_data['image'], tf.float32)
    train_data['image'] /= tf.constant(255.)
    session.run(test_data_iterator.initializer)
    test_data = test_data_iterator.get_next()
    test_data['image'] = tf.cast(test_data['image'], tf.float32)
    test_data['image'] /= tf.constant(255.)

    # create boosted model and connect to training data
    model = SequentialResNet(num_classes, num_channels)
    model(train_data['image'], is_train=True)

    # create optimizer, session, and partially evaluate `train`
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_module = functools.partial(
        train,
        session=session,
        train_data_iterator=train_data_iterator,
        train_batches_per_epoch=train_batches_per_epoch,
        test_data_iterator=test_data_iterator,
        test_batches_per_epoch=test_batches_per_epoch,
        epochs=epochs_per_module)

    # add module loop
    train_ops = []
    train_verbose_dicts = []
    test_verbose_dicts = []
    for _ in range(module_num):
        # add module and get train_op
        module = ResidualBoostingModule(repr_module_name,
                                        hypothesis_module_name, num_channels,
                                        num_classes)
        alpha, hypothesis, boosted_classification = model.add_module(
            module, optimizer, session)
        module_loss = model.get_hypothesis_loss(alpha, hypothesis,
                                                train_data['label'])
        module_train_op = module.get_train_op(optimizer, module_loss)
        train_ops.append(module_train_op)

        # calculate training values
        train_final_accuracy = calculate_accuracy(boosted_classification,
                                                  train_data['label'])
        train_verbose_dict = {
            'module_loss': module_loss,
            'final_accuracy': train_final_accuracy
        }
        train_verbose_dicts.append(train_verbose_dict)

        # calculate test values
        test_final_logits, test_hypotheses, _ = model(test_data['image'])
        test_module_loss = model.get_hypothesis_loss(alpha,
                                                     test_hypotheses[-1],
                                                     test_data['label'])
        test_accuracy = calculate_accuracy(test_final_logits,
                                           test_data['label'])
        test_verbose_dict = {
            'module_loss': test_module_loss,
            'final_accuracy': test_accuracy
        }
        test_verbose_dicts.append(test_verbose_dict)

    # train module loop
    session.run(tf.initializers.global_variables())
    for module_train_op, train_verbose_dict, test_verbose_dict in zip(
            train_ops, train_verbose_dicts, test_verbose_dicts):
        # run module training
        train_module(train_op=module_train_op,
                     train_verbose_dict=train_verbose_dict,
                     test_verbose_dict=test_verbose_dict)

    return model
