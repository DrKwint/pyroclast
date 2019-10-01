import sonnet as snt
import tensorflow as tf

mapping = {}


def register(name):

    def _thunk(func):
        mapping[name] = func
        return func

    return _thunk


@register('conv_block')
def conv_block(channels):
    return snt.Sequential([
        snt.Conv2D(channels, 3), tf.nn.relu,
        snt.Conv2D(channels, 3), tf.nn.relu
    ])


@register('linear_classifier')
def linear(classes):
    return snt.Linear(classes)


def get_module(name):
    """
    If you want to register your own module outside modules.py, you just need:

    Usage Examplee:
    -------------
    from pyroclast.selfboosting.modules import register
    @register("your_module_name")
    def your_module_define(**module_kwargs):
        ...
        return module_fn

    """
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown module: {}'.format(name))
