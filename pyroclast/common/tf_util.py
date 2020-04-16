import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
import tensorflow_datasets as tfds

import tensorflow_datasets as tfds

tfd = tfp.distributions
tfb = tfp.bijectors


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
        self.sum_product = np.array(0., dtype=np.float64)
        self.sum_first = np.array(0., dtype=np.float64)
        self.sum_second = np.array(0., dtype=np.float64)
        self.sum_sq_first = np.array(0., dtype=np.float64)
        self.sum_sq_second = np.array(0., dtype=np.float64)

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


def load_model(module,
               model_save_name,
               data_dict,
               output_dir=__file__ + '/../../../',
               conv_stack_name='ross_net',
               learning_rate=1e-4):
    build_savable_objects_func = getattr(module, 'build_savable_objects')
    if build_savable_objects_func is None:
        raise Exception(
            "Given module does not have a `build_savable_objects` method.")
    objects = build_savable_objects_func(conv_stack_name, data_dict,
                                         learning_rate, output_dir,
                                         model_save_name)

    if objects['ckpt_manager'].latest_checkpoint is not None:
        objects['checkpoint'].restore(
            objects['ckpt_manager'].latest_checkpoint).expect_partial()
    else:
        print("Wrong directory?")
    return objects['model']
