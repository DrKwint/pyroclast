import tensorflow as tf
import sonnet as snt

from pyroclast.common.models import mapping

mapping = {}


def register(name):

    def _thunk(func):
        mapping[name] = func
        return func

    return _thunk


@register("mnist_conv_encoder")
def mnist_conv(base_filters, **kwargs):
    """NHWC input should have two 2's in the prime factorization of its HW, outputs N,H/4,W/4,bf*2"""
    layers = []
    layers.append(
        tf.keras.layers.Conv2D(base_filters,
                               3,
                               2,
                               padding='SAME',
                               activation=tf.nn.elu))  # 14 x 14 x bf
    layers.append(
        tf.keras.layers.Conv2D(base_filters,
                               3,
                               padding='SAME',
                               activation=tf.nn.elu))
    layers.append(
        tf.keras.layers.Conv2D(base_filters * 2,
                               3,
                               2,
                               padding='SAME',
                               activation=tf.nn.elu))  # 7 x 7 x bf * 2
    layers.append(
        tf.keras.layers.Conv2D(base_filters * 2,
                               3,
                               padding='SAME',
                               activation=tf.nn.elu))
    return tf.keras.Sequential(layers)


@register("mnist_conv_decoder")
def mnist_deconv(base_filters, **kwargs):
    """Inputs NHWC, outputs N,H*4,W*4,output_channels"""
    layers = []
    layers.append(
        tf.keras.layers.Conv2DTranspose(filters=base_filters * 2,
                                        kernel_size=3,
                                        strides=2,
                                        padding='SAME',
                                        activation=tf.nn.elu))
    layers.append(
        tf.keras.layers.Conv2D(filters=base_filters * 2,
                               kernel_size=3,
                               padding='SAME',
                               activation=tf.nn.elu))
    layers.append(
        tf.keras.layers.Conv2DTranspose(filters=base_filters,
                                        kernel_size=3,
                                        strides=2,
                                        padding='SAME',
                                        activation=tf.nn.elu))
    layers.append(
        tf.keras.layers.Conv2D(filters=base_filters,
                               kernel_size=3,
                               padding='SAME',
                               activation=tf.nn.elu))
    if 'output_channels' in kwargs:
        layers.append(
            tf.keras.layers.Conv2D(filters=kwargs['output_channels'],
                                   kernel_size=1,
                                   name='dec_out'))
    return tf.keras.Sequential(layers)


def get_network_builder(name):
    """
    If you want to register your own network outside models.py, you just need:

    Usage Examplee:
    -------------
    from pyroclast.common.models import register
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
