import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

mapping = {}


def register(name):
    def _thunk(func):
        mapping[name] = func
        return func

    return _thunk


class Upscale(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(Upscale, self).__init__()

    def build(self, input_shape):
        n, h, w, c = input_shape
        self.transform = lambda x: tf.image.resize_nearest_neighbor(x, [2 * h, 2 * w])

    def call(self, input):
        return self.transform(input)


@register("upscale_conv")
def upscale_conv(num_blocks=8, init_filter_num=256, **network_kwargs):
    blocklist = [
        tf.keras.layers.Conv2D(
            init_filter_num, 3, padding='same', activation=tf.nn.leaky_relu),
        tf.keras.layers.Conv2D(
            init_filter_num, 3, padding='same', activation=tf.nn.leaky_relu),
    ]
    upscale_block = lambda num_filters: [
        tf.keras.layers.UpSampling2D(),
        tf.keras.layers.Conv2D(num_filters, 3, padding='same', activation=tf.nn.leaky_relu),
        tf.keras.layers.Conv2D(num_filters, 3, padding='same', activation=tf.nn.leaky_relu)]
    num_filters = init_filter_num
    for _ in range(num_blocks):
        blocklist += upscale_block(num_filters)
        num_filters //= 2
    net = tf.keras.Sequential(blocklist)
    return net


@register("mlp")
def mlp(output_sizes=[64] * 2, **network_kwargs):
    net = snt.nets.MLP(output_sizes, **network_kwargs)
    return lambda x: net(x)


@register("conv_only")
def conv_only(output_channels=[32, 64, 64, 128, 128],
              kernel_shapes=(5, 3, 3, 3, 3),
              strides=(4, 2, 1, 1, 1),
              **conv_kwargs):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            oc, ks, strides, activation=tf.nn.leaky_relu, padding='same')
        for (oc, ks, strides) in zip(output_channels, kernel_shapes, strides)
    ])


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
