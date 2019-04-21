import multiprocessing
import sys
from importlib import import_module

import tensorflow as tf

import pyroclast
from pyroclast.common.tf_util import get_session
from pyroclast.common.data_registry import get_dataset, split_dataset, dataset_tensor
from pyroclast.common.cmd_util import common_arg_parser, parse_unknown_args


def get_task_type(task_id):
    tasks = pyroclast.common.data_registry.tasks

    task_type = None
    for t, ids in tasks.items():
        if task_id in ids:
            task_type = t
            break
    assert task_type is not None, 'task_id {} is not recognized in task types'.format(
        task_id)

    return task_type, task_id


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
    # ncpu = multiprocessing.cpu_count()
    # if sys.platform == 'darwin': ncpu //= 2
    # alg = args.alg
    # seed = args.seed

    dataset = get_dataset(args.dataset)
    ds_t = lambda ds: dataset_tensor(ds, args.epochs, args.batch_size)
    if args.task == 'semisupervised':
        assert args.num_labeled is not None
        train_data = dataset()['train']
        labeled, unlabeled = split_dataset(train_data, args.num_labeled)
        labeled = ds_t(labeled)
        unlabeled = ds_t(unlabeled)
        train_data = (labeled, unlabeled)
    else:
        train_data = dataset()['train']
        train_data = ds_t(train_data)

    return train_data


def train(args, extra_args):
    # load data
    task_type, data_name = get_task_type(args.dataset)
    print('task_type: {}'.format(task_type))

    seed = args.seed
    train_data = setup_data(args)

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, task_type)
    alg_kwargs.update(extra_args)

    print('Training {} on {} with arguments \n{}'.format(
        args.alg, data_name, alg_kwargs))

    model = learn(train_data=train_data, seed=seed, **alg_kwargs)

    return model


def get_default_network(task_type):
    return 'cnn'


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


def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    model = train(args, extra_args)


if __name__ == '__main__':
    main(sys.argv)
