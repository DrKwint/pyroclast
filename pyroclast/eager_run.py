import sys
from importlib import import_module
from pyroclast.common.cmd_util import common_arg_parser, parse_unknown_args

import tensorflow as tf
tf.enable_eager_execution()


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


def train(args, extra_args):
    # load data
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, args.dataset)
    alg_kwargs.update(extra_args)

    print('Training {} on {} with arguments \n{}'.format(
        args.alg, args.dataset, alg_kwargs))

    model = learn(args.dataset, seed=seed, **alg_kwargs)

    return model


def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    model = train(args, extra_args)


if __name__ == '__main__':
    main(sys.argv)
