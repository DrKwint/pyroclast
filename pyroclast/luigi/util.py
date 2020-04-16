import configparser
import functools

import tensorflow as tf


@functools.lru_cache(maxsize=2)
def get_base_path(section):
    parser = configparser.ConfigParser()
    config_path = r'./local.cfg'
    parser.read(config_path)
    return parser.get(section, 'base_path')


def limit_gpu_memory_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus),
                      "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
