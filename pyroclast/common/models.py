import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.layers as layers

mapping = {}


def register(name):
    def _thunk(func):
        mapping[name] = func
        return func

    return _thunk


class UpscaleBlock(snt.AbstractModule):
    def __init__(self, num_filters, name='upscale_block'):
        super(UpscaleBlock, self).__init__(name=name)
        self._num_filters = num_filters

    def _build(self, inputs):
        x = inputs
        n, w, h, c = x.get_shape().as_list()
        net = snt.Sequential([
            lambda x: tf.image.resize_nearest_neighbor(x, [2 * h, 2 * w]),
            snt.Conv2D(self._num_filters, 3, padding=snt.SAME),
            tf.nn.leaky_relu,
            snt.Conv2D(self._num_filters, 3, padding=snt.SAME),
            tf.nn.leaky_relu,
        ])
        return net(x)


@register("upscale_conv")
def upscale_conv(num_blocks=5, init_filter_num=256, **network_kwargs):
    blocklist = [
        snt.Conv2D(init_filter_num, 4, padding=snt.SAME),
        tf.nn.leaky_relu,
        snt.Conv2D(init_filter_num, 3, padding=snt.SAME),
        tf.nn.leaky_relu,
    ]
    num_filters = init_filter_num
    for _ in range(num_blocks):
        blocklist.append(UpscaleBlock(num_filters))
        num_filters /= 2
    return snt.Sequential(blocklist)


@register("mlp")
def mlp(output_sizes=[64] * 2, **network_kwargs):
    return snt.nets.MLP(output_sizes, **network_kwargs)


@register("conv_only")
def conv_only(output_channels=[32, 64, 64],
              kernel_shapes=(5, 3, 3),
              strides=(4, 2, 1),
              **conv_kwargs):
    return snt.nets.ConvNet2D(
        output_channels,
        kernel_shapes,
        strides,
        paddings=[snt.SAME],
        activation=tf.nn.relu,
        activate_final=True)


def get_network_builder(name):
    """
    If you want to register your own network outside models.py, you just need:

    Usage Examplee:
    -------------
    from baselines.common.models import register
    @register("your_network_name")
    def your_network_define(**net_kwargs):
        ...
        return network_fn

    """
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown network type: {}'.format(name))
