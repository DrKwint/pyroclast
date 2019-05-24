import sonnet as snt
import tensorflow as tf
import numpy as np

from pyroclast import selfboosting
from pyroclast.selfboosting.build_graph import SequentialResNet
from pyroclast.common.tf_util import run_epoch_ops, calculate_accuracy


def learn(train_data_iterator,
          train_batches_per_epoch,
          test_data_iterator,
          test_batches_per_epoch,
          seed,
          epochs=10,
          class_num=10,
          module_num=10,
          module_name='conv_block',
          representation_channels=32):
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

    # build boosted model and get the first module
    model = SequentialResNet(class_num, representation_channels)
    module = model.boosting_modules[-1]

    # connect train model
    train_logits, train_hypotheses, _ = model(train_data['image'])
    train_loss = model.get_hypothesis_loss(module.alpha, train_hypotheses[-1],
                                           train_data['label'])
    train_accuracy = calculate_accuracy(train_logits, train_data['label'])

    # connect test model
    test_logits, test_hypotheses, _ = model(test_data['image'])
    test_loss = model.get_hypothesis_loss(module.alpha, test_hypotheses[-1],
                                          test_data['label'])
    test_accuracy = calculate_accuracy(test_logits, test_data['label'])

    # define optimizer and train operation
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = model.get_module_train_op(optimizer, module, train_loss)

    # initialize parameters and train
    session.run(tf.initializers.global_variables())
    for epoch in range(epochs):
        # re-initialize data iterators
        session.run(train_data_iterator.initializer)
        session.run(test_data_iterator.initializer)

        # run a training epoch
        print("Epoch ", epoch)
        train_vals = run_epoch_ops(
            session,
            train_batches_per_epoch, {
                'loss': train_loss,
                'accuracy': train_accuracy,
            }, [train_op],
            verbose=True)

        # log train epoch info
        print("train loss mean: ", np.mean(train_vals['loss']),
              ', train loss median: ', np.median(train_vals['loss']))
        print("train accuracy: ", np.mean(train_vals['accuracy']))

        # run a test epoch
        test_vals = run_epoch_ops(
            session,
            test_batches_per_epoch, {
                'loss': test_loss,
                'accuracy': test_accuracy
            },
            verbose=True)

        # log test epoch info
        print("test loss mean: ", np.mean(test_vals['loss']),
              ', test loss median: ', np.median(test_vals['loss']))
        print("test accuracy: ", np.mean(test_vals['accuracy']))

    return model
