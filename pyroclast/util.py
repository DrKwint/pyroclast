from pyroclast.common.datasets import check_datasets, get_dataset_builder
from pyroclast.common.tf_util import setup_tfds
from pathlib import Path


def direct(func):

    def wrapper(*args, **kwargs):
        dataset = kwargs['data_dict']
        batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 32
        resize_data_shape = kwargs[
            'resize_data_shape'] if 'resize_data_shape' in kwargs else None
        data_limit = kwargs['data_limit'] if 'data_limit' in kwargs else -1
        data_dir = kwargs['data_dir'] if 'data_dir' in kwargs else None
        seed = kwargs['seed'] if 'seed' in kwargs else None
        output_dir = kwargs['output_dir'] if 'output_dir' in kwargs else './'
        if check_datasets(dataset):
            data_dict = get_dataset_builder(dataset)(batch_size,
                                                     resize_data_shape,
                                                     data_limit, data_dir)
        else:
            data_dict = setup_tfds(dataset,
                                   batch_size,
                                   resize_data_shape,
                                   data_limit,
                                   data_dir,
                                   shuffle_seed=seed)

        kwargs['data_dict'] = data_dict
        func(*args, **kwargs)
        (Path(output_dir) / 'done').touch()

    return wrapper