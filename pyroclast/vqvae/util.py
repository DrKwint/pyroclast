import sonnet as snt
import tensorflow as tf
from tqdm import tqdm
import numpy as np


def residual_block(channels):
    return snt.Sequential([
        snt.Conv2D(channels, 3),
        tf.nn.relu,
        snt.Conv2D(channels, 1),
        tf.nn.relu,
    ])


class NIN(tf.Module):
    """ a network in network layer (1x1 CONV) """

    def __init__(self, num_units, **kwargs):
        self.num_units = num_units
        self.dense = snt.Linear(num_units, **kwargs)

    def __call__(self, x):
        s = int_shape(x)
        x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
        x = self.dense(x)
        return tf.reshape(x, s[:-1] + [self.num_units])


def int_shape(x):
    return list(map(int, x.get_shape()))


def down_shift(x, step=1):
    xs = int_shape(x)
    return tf.concat(
        [tf.zeros([xs[0], step, xs[2], xs[3]]), x[:, :xs[1] - step, :, :]], 1)


def right_shift(x, step=1):
    xs = int_shape(x)
    return tf.concat(
        [tf.zeros([xs[0], xs[1], step, xs[3]]), x[:, :, :xs[2] - step, :]], 2)


def left_shift(x, step=1):
    xs = int_shape(x)
    return tf.concat([
        x[:, :, step:, :],
        tf.zeros([xs[0], xs[1], step, xs[3]]),
    ], 2)


def gated_activation(outputs):
    depth = outputs.shape[-1] // 2
    tanh = tf.nn.tanh(outputs[:, :, :, :depth])
    sigmoid = tf.nn.sigmoid(outputs[:, :, :, depth:])
    return tanh * sigmoid


def load_args_from_dir(dir_):
    from pathlib import Path
    import json
    with open(Path(dir_) / 'parameters.json') as param_json:
        loaded_args = json.load(param_json)
    args = loaded_args['args']
    args.update(loaded_args['module_kwargs'])
    return args


def train(data_dict,
          train_minibatch_fn,
          eval_minibatch_fn,
          early_stopping,
          output_log_file,
          debug=False):
    # run training loop
    train_batches = data_dict['train']
    test_batches = data_dict['test']
    if debug:
        train_batches = tqdm(train_batches, total=data_dict['train_bpe'])
        test_batches = tqdm(test_batches, total=data_dict['test_bpe'])
    epoch = 0
    while True:
        epoch += 1
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

        # save parameters and do early stopping
        if early_stopping(epoch, float(sum_loss) / float(sum_num_samples)):
            break
