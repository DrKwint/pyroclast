import functools
import multiprocessing
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import tqdm


def setup_tfds(dataset,
               batch_size,
               resize_data_shape=None,
               data_limit=-1,
               data_dir=None,
               shuffle_seed=None):
    """Setup a TensorFlow Dataset

    Args:
        dataset (str): Name of the TFDS to load `(Full list here) <https://www.tensorflow.org/datasets/catalog/overview>`_
        batch_size (int): Number of data per batch
        resize_data_shape (interable of int): If specified, do an image reshape on the data to the given shape
        data_limit (int): Upper limit to the number of data to load, -1 to load all data
        data_dir (relative path str): Directory where TFDS should look for the data files

    Returns:
        A dict with keys train, test, train_bpe, test_bpe, shape, and num_classes.

        train, test are iterators over batches of each set respectively.
        train_bpe and test_bpe are the number of batches per epoch in each set
        shape is the shape of each datum
        num_classes is the number of classes in the labels
    """
    data_dict, info = tfds.load(dataset, with_info=True, data_dir=data_dir)
    data_dict['name'] = dataset
    data_dict['train_bpe'] = info.splits['train'].num_examples // batch_size
    data_dict['train_num'] = info.splits['train'].num_examples
    data_dict['test_bpe'] = info.splits['test'].num_examples // batch_size
    data_dict['test_num'] = info.splits['test'].num_examples
    data_dict['num_classes'] = info.features['label'].num_classes

    def resize_ds_img(features):
        features['image'] = tf.image.resize(features['image'],
                                            resize_data_shape)
        # I'm not actually sure if this bit with the mask is right at all, but it's needed for batching right now
        if 'segmentation_mask' in features:
            features['segmentation_mask'] = tf.image.resize(
                features['segmentation_mask'], resize_data_shape)
        return features

    if resize_data_shape is None:
        data_dict['shape'] = info.features['image'].shape
    else:
        data_dict['train'] = data_dict['train'].map(resize_ds_img)
        data_dict['test'] = data_dict['test'].map(resize_ds_img)
        data_dict['shape'] = resize_data_shape + [
            info.features['image'].shape[-1]
        ]

    data_dict['train'] = data_dict['train'].shuffle(
        1024, seed=shuffle_seed).take(data_limit).batch(batch_size).prefetch(
            tf.data.experimental.AUTOTUNE)
    data_dict['test'] = data_dict['test'].shuffle(
        1024, seed=shuffle_seed).take(data_limit).batch(batch_size).prefetch(
            tf.data.experimental.AUTOTUNE)
    return data_dict


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
                k: v + [np.array(out[k])] for k, v in verbose_vals.items()
            }
        except tf.errors.OutOfRangeError:
            break

    return {
        k: np.stack(v) if v is not None else np.array()
        for k, v in verbose_vals.items()
    }


class DiscretizedLogistic(tfp.distributions.Distribution):

    def __init__(self,
                 loc,
                 log_scale=None,
                 event_dims=3,
                 dtype=tf.float32,
                 validate_args=True,
                 allow_nan_stats=False,
                 name="DiscretizedLogistic"):
        super(DiscretizedLogistic,
              self).__init__(dtype, tfp.distributions.NOT_REPARAMETERIZED,
                             validate_args, allow_nan_stats)
        self._dtype = dtype
        self._loc = loc
        self._event_dims = event_dims
        self._batch_dims = loc.shape.ndims - event_dims
        self._scale = tfp.util.DeferredTensor(
            tf.math.exp,
            tf.Variable(tf.zeros(self._loc.shape[-event_dims:]),
                        name='LogScale'))

    def _event_shape(self):
        return tf.shape(self._loc)[-self._event_dims:]

    def _batch_shape(self):
        return tf.shape(self._loc)[:self._batch_dims]

    def _mean(self):
        return self._loc

    def _log_prob(self, sample, binsize=1 / 256.0):
        mean = self._loc
        sample = (tf.floor(sample / binsize) * binsize - mean) / self._scale
        logp = tf.math.log(
            tf.sigmoid(sample + binsize / self._scale) - tf.sigmoid(sample) +
            1e-7)
        return tf.reduce_sum(
            logp, range(self._batch_dims, self._batch_dims + self._event_dims))


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keepdims=True)
    return m + tf.math.log(tf.reduce_sum(tf.exp(x - m2), axis))


def img_discretized_logistic_log_prob(mean,
                                      sample,
                                      log_scale,
                                      binsize=1 / 256.0,
                                      log_scale_min=-7.):
    log_scale = tf.maximum(log_scale, log_scale_min)

    centered_sample = sample - mean
    inv_stdv = tf.exp(-log_scale)
    plus_in = inv_stdv * (centered_sample + binsize)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_sample - binsize)
    cdf_min = tf.nn.sigmoid(min_in)

    log_cdf_plus = plus_in - tf.nn.softplus(
        plus_in)  # log probability for edge case of 0
    log_one_minus_cdf_min = -tf.nn.softplus(
        min_in)  # log probability for edge case of 1
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases

    #log probability in the center of the bin, to be used in extreme cases
    mid_in = inv_stdv * centered_sample
    log_pdf_mid = mid_in - log_scale - 2. * tf.nn.softplus(mid_in)

    logp = tf.where(
        sample < -0.999, log_cdf_plus,
        tf.where(
            sample > 0.999, log_one_minus_cdf_min,
            tf.where(cdf_delta > 1e-5,
                     tf.math.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid)))

    if not tf.reduce_all(tf.math.is_finite(mean)):
        print("mean ISN'T FINITE")
    if not tf.reduce_all(tf.math.is_finite(log_scale)):
        print("log_scale ISN'T FINITE")
    if not tf.reduce_all(tf.math.is_finite(inv_stdv)):
        print("inv_stdv ISN'T FINITE")
    if not tf.reduce_all(tf.math.is_finite(sample)):
        print("sample ISN'T FINITE")
    if not tf.reduce_all(tf.math.is_finite(cdf_plus)):
        print("cdf_plus ISN'T FINITE")
    if not tf.reduce_all(tf.math.is_finite(cdf_min)):
        print("cdf_min ISN'T FINITE")
    if not tf.reduce_all(tf.math.is_finite(logp)):
        print("LOGP ISN'T FINITE")

    #sample = (tf.floor(sample / binsize) * binsize - mean) / scale
    #logp = tf.math.log(
    #    tf.sigmoid(sample + binsize / scale) - tf.sigmoid(sample) + 1e-7)
    return tf.reduce_sum(log_sum_exp(logp), [1, 2])


def correlation(a, b):
    mean_a = tf.math.reduce_mean(a)
    mean_b = tf.math.reduce_mean(b)

    deviations_a = a - mean_a
    deviations_b = b - mean_b

    numerator = tf.math.reduce_sum(deviations_a * deviations_b)
    denominator = tf.math.sqrt(tf.reduce_sum(
        tf.math.square(deviations_a))) * tf.math.sqrt(
            tf.reduce_sum(tf.math.square(deviations_b)))
    return numerator / (denominator + 1e-12)


class OnePassCorrelation(object):

    def __init__(self):
        self.n = 0
        self.sum_product = 0.
        self.sum_first = 0.
        self.sum_second = 0.
        self.sum_sq_first = 0.
        self.sum_sq_second = 0.

    def accumulate(self, a, b):
        self.n += a.shape[0]
        self.sum_product += tf.reduce_sum(a * b, axis=0)
        self.sum_first += tf.reduce_sum(a, axis=0)
        self.sum_second += tf.reduce_sum(b, axis=0)
        self.sum_sq_first += tf.reduce_sum(a**2, axis=0)
        self.sum_sq_second += tf.reduce_sum(b**2, axis=0)

    def finalize(self):
        numerator = (self.n * self.sum_product) - (self.sum_first *
                                                   self.sum_second)
        denominator = tf.math.sqrt(self.n * self.sum_sq_first - tf.math.square(
            self.sum_first)) * tf.math.sqrt(self.n * self.sum_sq_second -
                                            tf.math.square(self.sum_second))
        return numerator / (denominator + 1e-12)
