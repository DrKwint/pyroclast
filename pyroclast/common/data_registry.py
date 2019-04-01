import functools

import tensorflow as tf
import tensorflow_datasets as tfds

DOWNLOAD_LOCATION = '../tensorflow_datasets'

mapping = {}
tasks = {}


def register(name):
    def _thunk(func):
        mapping[name] = func
        return func

    return _thunk


def task(task, name):
    singleton = set([name])
    try:
        tasks[task] = tasks[task].union(singleton)
    except KeyError:
        tasks[task] = singleton

    def passthrough(func):
        return func

    return passthrough


@register("mnist")
@task("classification", "mnist")
def mnist(**dataset_kwargs):
    return tfds.load('mnist', data_dir=DOWNLOAD_LOCATION)


@register("cifar10")
@task("classification", "cifar10")
def cifar10(**dataset_kwargs):
    return tfds.load('cifar10', data_dir=DOWNLOAD_LOCATION)


def split_dataset(dataset, labeled_num):
    return dataset.take(labeled_num), dataset.skip(labeled_num)


def dataset_tensor(ds, epochs, batch_size, buffer_size=1000):
    ds = ds.shuffle(buffer_size)
    ds = ds.repeat(epochs)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size)
    return tf.data.make_one_shot_iterator(ds).get_next()


def get_dataset(name):
    """
    If you want to register your own dataset outside data_registry.py, you just need:

    Usage Examplee:
    -------------
    from pyroclast.common.data_registry import register
    @register("your_dataset_name")
    def your_dataset_define(**dataset_kwargs):
        ...
        return dataset_fn

    """
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown dataset: {}'.format(name))
