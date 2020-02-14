import json
import os
import sys
from importlib import import_module

import tensorflow as tf
from tensorflow.python.client import device_lib

from pyroclast.common.cmd_util import common_arg_parser, parse_unknown_args
from pyroclast.common.tf_util import setup_tfds
from pyroclast.common.datasets import check_datasets, get_dataset_builder

tf.compat.v1.enable_eager_execution()


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


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['pyroclast', alg, submodule]))
    except ImportError:
        print('failed to import from {}'.format('.'.join(['pyroclast', alg])))
        assert False

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, dataset):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, dataset)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def train(args, extra_args):
    # load data
    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, args.dataset)
    alg_kwargs.update(extra_args)
    if check_datasets(args.dataset):
        data_dict = get_dataset_builder(args.dataset)(args.batch_size,
                                                      args.resize_data_shape,
                                                      args.data_limit,
                                                      args.data_dir)
    else:
        data_dict = setup_tfds(args.dataset, args.batch_size,
                               args.resize_data_shape, args.data_limit,
                               args.data_dir)

    print('Training {} on {} with arguments \n{}'.format(
        args.alg, args.dataset, alg_kwargs))
    model = learn(data_dict,
                  seed=args.seed,
                  output_dir=args.output_dir,
                  debug=args.debug,
                  **alg_kwargs)

    return model


def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

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

    model = train(args, extra_args)


if __name__ == '__main__':
    main(sys.argv)
