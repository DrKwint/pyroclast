import multiprocessing
import os

import numpy as np
import tensorflow as tf
import tqdm


def calculate_accuracy(logits, label):
    """Compare argmax logits to int label, returns value in [0,1]"""
    prediction = tf.argmax(logits, 1)
    equality = tf.equal(prediction, label)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy


def get_session(config=None):
    """Get default session or create one with a given config"""
    sess = tf.get_default_session()
    if sess is None:
        sess = make_session(config=config, make_default=True)
    return sess


def make_session(config=None, num_cpu=None, make_default=False, graph=None):
    """Returns a session that will use <num_cpu> CPU's only"""
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    if config is None:
        config = tf.ConfigProto(allow_soft_placement=True,
                                inter_op_parallelism_threads=num_cpu,
                                intra_op_parallelism_threads=num_cpu)
        config.gpu_options.allow_growth = True

    if make_default:
        return tf.InteractiveSession(config=config, graph=graph)
    else:
        return tf.Session(config=config, graph=graph)


def run_epoch_ops(session,
                  steps_per_epoch,
                  verbose_ops_dict=None,
                  silent_ops=None,
                  feed_dict_fn=lambda: None,
                  verbose=False):
    """
    Args:
        session (tf.Session): Session with tf.Graph containing the operations
            passed in `verbose_ops_dict` and `silent_ops`
        steps_per_epoch (int): number of times to run operations
        verbose_ops_dict (dict): strings to tf operations whose values will be
            returned
        silent_ops (list): list of tf operations to run, ignoring output
        feed_dict_fn (callable): called to retrieve the feed_dict
            (dict of tf.placeholder to np.array)
        verbose (bool): whether to use tqdm progressbar on stdout
    Return:
        dict of str to np.array parallel to the verbose_ops_dict
    """
    if verbose_ops_dict is None: verbose_ops_dict = dict()
    if silent_ops is None: silent_ops = list()
    verbose_vals = {k: [] for k, v in verbose_ops_dict.items()}
    if verbose:
        iterable = tqdm.tqdm(list(range(steps_per_epoch)))
    else:
        iterable = list(range(steps_per_epoch))
    for _ in iterable:
        try:
            out = session.run([silent_ops, verbose_ops_dict],
                              feed_dict=feed_dict_fn())[1]

            verbose_vals = {
                k: v + [np.array(out[k])]
                for k, v in verbose_vals.items()
            }
        except tf.errors.OutOfRangeError:
            break

    return {
        k: np.stack(v) if v is not None else np.array()
        for k, v in verbose_vals.items()
    }
