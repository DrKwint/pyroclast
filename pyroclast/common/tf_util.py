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
        config = tf.ConfigProto(
            allow_soft_placement=True,
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
            out = session.run(
                [silent_ops, verbose_ops_dict], feed_dict=feed_dict_fn())[1]

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


import sonnet as snt
import tensorflow as tf

tfd = tf.contrib.distributions


class DiscretizedLogistic(tf.keras.Model, tfd.Distribution):
    def __init__(self, loc, batch_dims=3, name="discretized_logistic"):
        super(DiscretizedLogistic, self).__init__(name=name)
        self._name = name
        self._dtype = tf.float32
        self._reparameterization_type = tfd.NOT_REPARAMETERIZED
        self._allow_nan_stats = False
        self._graph_parents = []
        self._loc = loc
        self._batch_dims = batch_dims
        self._log_scale = tf.get_variable(
            "log_scale",
            initializer=tf.zeros(loc.get_shape().as_list()[-batch_dims:]),
            dtype=tf.float32)

    def mean(self):
        return self._loc

    @property
    def scale(self):
        return tf.exp(self._log_scale)

    def log_prob(self, sample, binsize=1 / 256.0):
        scale = tf.exp(self._log_scale)
        mean = self._loc
        sample = (tf.floor(sample / binsize) * binsize - mean) / scale
        logp = tf.log(
            tf.sigmoid(sample + binsize / scale) - tf.sigmoid(sample) + 1e-7)
        return logp  # tf.reduce_sum(logp, [2, 3, 4])
