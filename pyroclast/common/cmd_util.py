import argparse


def parse_unknown_args(args):
    """
    Parse arguments not consumed by arg parser into a dicitonary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval


def common_arg_parser():
    """
    Create an argparse.ArgumentParser.
    """
    parser = arg_parser()
    parser.add_argument('--task',
                        '-t',
                        help='Task to run',
                        type=str,
                        default='learn')
    parser.add_argument('--dataset', help='dataset', type=str, default='mnist')
    parser.add_argument('--module',
                        help="""The module with the
                        given task. Actually resolves to pyroclast.module.module to
                        search for the task.""",
                        type=str)
    parser.add_argument('--submodule',
                        help="""The submodule used. Changes the resolution to
                        pyroclast.module.submodule. pyroclast.module.defaults is still
                        used.""",
                        type=str,
                        default=None)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--resize_data_shape',
                        type=int,
                        nargs='+',
                        default=None)
    parser.add_argument('--data_limit', type=int, default=-1)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--debug', action='store_true')
    return parser


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    return argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
