import json
import os
import sys
from importlib import import_module

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.client import device_lib

from pyroclast.common.cmd_util import common_arg_parser, parse_unknown_args
from pyroclast.common.util import img_preprocess

print(device_lib.list_local_devices())
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
        alg_module = import_module('.'.join(['pyroclast',
                                             alg]))  #, submodule]))
    except ImportError:
        # then from rl_algs
        # alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))
        print('failed to import from {}'.format('.'.join(['pyroclast', alg])))
        assert False

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def setup_data(args):
    # load data
    data_dict, info = tfds.load(args.dataset,
                                with_info=True,
                                data_dir='./data/')
    if args.dataset == 'celeb_a':
        data_dict['num_classes'] = 2
        data_dict[
            'train_bpe'] = info.splits['train'].num_examples // args.batch_size
        data_dict[
            'test_bpe'] = info.splits['test'].num_examples // args.batch_size
        data_dict['shape'] = info.features['image'].shape

        data_dict['all_train'] = data_dict['train']
        data_dict['train'] = data_dict['train'].map(
            lambda x: img_preprocess(x, args.image_size)).shuffle(1024).batch(
                args.batch_size)
        data_dict['all_test'] = data_dict['test']
        data_dict['test'] = data_dict['test'].map(
            lambda x: img_preprocess(x, args.image_size)).batch(args.batch_size)
    else:
        data_dict[
            'train_bpe'] = info.splits['train'].num_examples // args.batch_size
        data_dict[
            'test_bpe'] = info.splits['test'].num_examples // args.batch_size
        data_dict['shape'] = info.features['image'].shape
        data_dict['num_classes'] = info.features['label'].num_classes

        data_dict['train'] = data_dict['train'].shuffle(1024).batch(
            args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        data_dict['test'] = data_dict['test'].batch(args.batch_size).prefetch(
            tf.data.experimental.AUTOTUNE)

    return data_dict


def train(args, extra_args):
    # load data
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, args.dataset)
    alg_kwargs.update(extra_args)
    data_dict = setup_data(args)

    print('Training {} on {} with arguments \n{}'.format(
        args.alg, args.dataset, alg_kwargs))
    model = learn(data_dict,
                  seed=seed,
                  output_dir=args.output_dir,
                  **alg_kwargs)

    return model


def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

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
