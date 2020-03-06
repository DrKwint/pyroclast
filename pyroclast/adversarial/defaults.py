import numpy as np


def mnist():
    return {
        'module_name': 'pyroclast.features.features',
        'model_name': 'mnist_basic',
        'norm': np.inf,
        'data_index': 0,
    }


def cifar10():
    return {
        'module_name': 'pyroclast.features.features',
        'model_name': 'basic_cifar_10',
        'norm': np.inf,
        'data_index': 0,
    }
