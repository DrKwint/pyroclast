import functools

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
        tasks[task] += singleton
    except KeyError:
        tasks[task] = singleton

    def passthrough(func):
        return func

    return passthrough


@register("mnist")
@task("classification", "mnist")
def mnist(**dataset_kwargs):
    return tfds.load('mnist', data_dir=DOWNLOAD_LOCATION)


def split_dataset(dataset, labeled_num):
    return dataset.take(labeled_num), dataset.skip(labeled_num)


def dataset_tensor(ds, epochs, batch_size, buffer_size=10000):
    ds = ds.repeat(epochs)
    ds = ds.shuffle(buffer_size)
    return ds.batch(batch_size)


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
