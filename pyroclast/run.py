import json
import os
import sys
from importlib import import_module
from pathlib import Path

import tensorflow as tf
from tensorflow.python.client import device_lib

from pyroclast.common.cmd_util import common_arg_parser, parse_unknown_args
from pyroclast.common.datasets import check_datasets, get_dataset_builder
from pyroclast.common.tf_util import setup_tfds

tf.compat.v1.enable_eager_execution()


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


limit_gpu_memory_usage()


def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary,
    evaluating python objects when possible
    '''

    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k, v in parse_unknown_args(args).items()}


def get_module(module_name, submodule=None):
    submodule = submodule or module_name
    try:
        # first try to import the module from baselines
        module = import_module('.'.join(['pyroclast', module_name]))
    except ImportError:
        print('failed to import from {}'.format('.'.join(
            ['pyroclast', module_name])))
        assert False

    return module


def get_task_function(module, func_name, submodule=None):
    return getattr(get_module(module, submodule), func_name)


def get_task_function_defaults(module_name, dataset):
    try:
        module_defaults = get_module(module_name, 'defaults')
        kwargs = getattr(module_defaults, dataset)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def run_task(args, extra_args):
    # load data
    task_func = get_task_function(args.module, args.task, args.submodule)
    module_kwargs = get_task_function_defaults(args.module, args.dataset)
    module_kwargs.update(extra_args)
    if check_datasets(args.dataset):
        data_dict = get_dataset_builder(args.dataset)(args.batch_size,
                                                      args.resize_data_shape,
                                                      args.data_limit,
                                                      args.data_dir)
    else:
        data_dict = setup_tfds(args.dataset,
                               args.batch_size,
                               args.resize_data_shape,
                               args.data_limit,
                               args.data_dir,
                               shuffle_seed=args.seed)

    print('Running {} on {} with arguments \n{}'.format(args.task, args.module,
                                                        args.dataset,
                                                        module_kwargs))
    model = task_func(data_dict,
                      seed=args.seed,
                      output_dir=args.output_dir,
                      debug=args.debug,
                      **module_kwargs)

    return model


def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if args.seed is not None:
        tf.random.set_seed(args.seed)

    if args.debug:
        print(device_lib.list_local_devices())

    # save the parameters
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, 'parameters.log'), 'w') as p_file:
        json.dump(
            {
                'args': args.__dict__,
                'unknown_args': unknown_args,
                'extra_args': extra_args
            }, p_file)

    model = run_task(args, extra_args)

    # signal completion
    (Path(args.output_dir) / 'done').touch()


if __name__ == '__main__':
    main(sys.argv)
