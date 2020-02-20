import copy
import os

import tensorflow as tf
from tqdm import tqdm

from pyroclast.common.early_stopping import EarlyStopping
from pyroclast.common.models import get_network_builder
from pyroclast.common.preprocessed_dataset import PreprocessedDataset
from pyroclast.common.util import dummy_context_mgr
from pyroclast.features.generic_classifier import GenericClassifier
from pyroclast.common.util import heatmap


# define minibatch fn
def run_minibatch(model,
                  optimizer,
                  global_step,
                  epoch,
                  batch,
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

    with tf.GradientTape() if is_train else dummy_context_mgr() as tape:
        global_step.assign_add(1)
        y_hat = model(x)
        classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=y_hat)

        # build loss calculate gradients and apply update
        mean_loss = tf.reduce_mean(classification_loss)
        if is_train:
            train_vars = model.trainable_variables
            gradients = tape.gradient(mean_loss, train_vars)
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
        tf.summary.scalar(prefix + "loss/mean final loss",
                          mean_loss,
                          step=global_step)
    loss_numerator = tf.reduce_sum(classification_loss)
    accuracy_numerator = tf.reduce_sum(
        tf.cast(tf.equal(prediction, labels), tf.int32))
    denominator = x.shape[0]
    return loss_numerator, accuracy_numerator, denominator


def train(data_dict, model, optimizer, global_step, writer, early_stopping,
          train_conv_stack, checkpoint, ckpt_manager, debug):
    if train_conv_stack:
        train_model = model
    else:
        train_model = model.classifier

    # use the max epoch value in early_stopping
    for epoch in range(10000):
        # train
        train_batches = data_dict['train']
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
          is_train=True,
          train_conv_stack=False,
          patience=2,
          max_epochs=10):
    writer = tf.summary.create_file_writer(output_dir)
    global_step = tf.compat.v1.train.get_or_create_global_step()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         beta_1=0.5,
                                         epsilon=0.01)

    conv_stack = tf.keras.applications.VGG19(
        include_top=False,
        weights='imagenet',
        input_shape=data_dict['shape'],
        pooling=None)  #get_network_builder(conv_stack)()
    classifier = tf.keras.Sequential(
        [tf.keras.layers.Dense(data_dict['num_classes'])])
    model = GenericClassifier(conv_stack, classifier)
    # setup checkpointing
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=model,
                                     global_step=global_step)
    ckpt_manager = tf.train.CheckpointManager(checkpoint,
                                              directory=os.path.join(
                                                  output_dir, 'phase1_model'),
                                              max_to_keep=3)

    for x in data_dict['train']:
        batch_size = x['label'].shape[0]
        break
    preprocessed_dataset = copy.copy(data_dict)
    preprocessed_dataset['train'] = PreprocessedDataset(
        data_dict['train'], model.features,
        'vgg19' + data_dict['name'] + '_train')(batch_size)
    preprocessed_dataset['test'] = PreprocessedDataset(
        data_dict['test'], model.features,
        'vgg19' + data_dict['name'] + '_test')(batch_size)

    if is_train:
        early_stopping = EarlyStopping(patience,
                                       ckpt_manager,
                                       eps=0.03,
                                       max_epochs=max_epochs)
        train(preprocessed_dataset, model, optimizer, global_step, writer,
              early_stopping, False, checkpoint, ckpt_manager, debug)
    usefulness = model.usefulness(preprocessed_dataset['test'],
                                  is_preprocessed=True)
    print(tf.reduce_min(usefulness), tf.reduce_mean(usefulness),
          tf.reduce_max(usefulness))
    heatmap(usefulness, 'rho_usefulness.png', 'rho usefulness')
