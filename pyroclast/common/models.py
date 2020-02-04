import numpy as np
import tensorflow as tf

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
        self.transform = lambda x: tf.image.resize_nearest_neighbor(
            x, [2 * h, 2 * w])

    def call(self, input):
        return self.transform(input)


@register("upscale_conv")
def upscale_conv(num_blocks=6, init_filter_num=256, **network_kwargs):
    blocklist = [
        tf.keras.layers.Conv2D(init_filter_num,
                               3,
                               padding='same',
                               activation=tf.nn.leaky_relu),
    ]
    upscale_block = lambda num_filters: [
        tf.keras.layers.UpSampling2D(),
        tf.keras.layers.Conv2D(
            num_filters, 3, padding='same', activation=tf.nn.leaky_relu),
        tf.keras.layers.Conv2D(
            num_filters, 3, padding='same', activation=tf.nn.leaky_relu)
    ]
    num_filters = init_filter_num
    for _ in range(num_blocks):
        blocklist += upscale_block(num_filters)
        num_filters //= 2
    net = tf.keras.Sequential(blocklist)
    return net


@register("conv_only")
def conv_only(output_channels=[32, 64, 64, 128],
              kernel_shapes=(5, 3, 3, 3),
              strides=(2, 2, 1, 1),
              **conv_kwargs):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(oc,
                               ks,
                               strides,
                               activation=tf.nn.leaky_relu,
                               padding='same')
        for (oc, ks, strides) in zip(output_channels, kernel_shapes, strides)
    ])


@register("mnist_encoder")
def mnist_conv():
    layers = []
    layers.append(
        tf.keras.layers.Conv2D(16, 3, 2, padding='SAME',
                               activation=tf.nn.relu))  # 14 x 14 x 16
    layers.append(
        tf.keras.layers.Conv2D(32, 3, 2, padding='SAME',
                               activation=tf.nn.relu))  # 7 x 7 x 32
    return tf.keras.Sequential(layers)


@register("mnist_decoder")
def mnist_deconv():
    layers = []
    layers.append(tf.keras.layers.Dense(49, activation=tf.nn.relu))
    layers.append(tf.keras.layers.Reshape(target_shape=[7, 7, 1]))
    layers.append(
        tf.keras.layers.Conv2DTranspose(filters=32,
                                        kernel_size=5,
                                        strides=2,
                                        padding='SAME',
                                        activation=tf.nn.relu))
    layers.append(
        tf.keras.layers.Conv2DTranspose(filters=16,
                                        kernel_size=5,
                                        strides=2,
                                        padding='SAME',
                                        activation=tf.nn.relu))
    return tf.keras.Sequential(layers)


@register("celeba_enc")
def celeba_conv(is_train=True):
    layers = []
    ef_dim = 64
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    # input.shape = (b_size,64,64,3)
    layers.append(
        tf.keras.layers.Conv2D(ef_dim, (5, 5), (2, 2),
                               padding='SAME',
                               kernel_initializer=w_init,
                               name='h0/conv2d'))
    layers.append(
        tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init,
                                           name='h0/batch_norm'))
    layers.append(tf.keras.layers.ReLU())
    # net_h0.outputs._shape = (b_size,32,32,64)

    layers.append(
        tf.keras.layers.Conv2D(ef_dim * 2, (5, 5), (2, 2),
                               padding='SAME',
                               kernel_initializer=w_init,
                               name='h1/conv2d'))
    layers.append(
        tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init,
                                           name='h1/batch_norm'))
    layers.append(tf.keras.layers.ReLU())
    # net_h1.outputs._shape = (b_size,16,16,128)

    layers.append(
        tf.keras.layers.Conv2D(ef_dim * 4, (5, 5), (2, 2),
                               padding='SAME',
                               kernel_initializer=w_init,
                               name='h2/conv2d'))
    layers.append(
        tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init,
                                           name='h2/batch_norm'))
    layers.append(tf.keras.layers.ReLU())
    # net_h2.outputs._shape = (b_size,8,8,256)

    layers.append(
        tf.keras.layers.Conv2D(ef_dim * 4, (5, 5), (2, 2),
                               padding='SAME',
                               kernel_initializer=w_init,
                               name='h3/conv2d'))
    layers.append(
        tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init,
                                           name='h3/batch_norm'))
    layers.append(tf.keras.layers.ReLU())
    # net_h2.outputs._shape = (b_size,4,4,512)

    layers.append(tf.keras.layers.Flatten(name='h4/flatten'))
    return tf.keras.Sequential(layers)


@register('celeba_dec')
def celeba_gen(is_train=True):
    layers = []
    gf_dim = 64
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    layers.append(
        tf.keras.layers.Dense(units=gf_dim * 4 * 16 * 16,
                              kernel_initializer=w_init,
                              name='o0/lin'))
    layers.append(
        tf.keras.layers.Reshape(target_shape=[16, 16, gf_dim * 4],
                                name='o0/reshape'))
    # 16, 16 = s16

    layers.append(
        tf.keras.layers.Conv2DTranspose(filters=gf_dim * 4,
                                        kernel_size=(5, 5),
                                        strides=(2, 2),
                                        padding='SAME',
                                        kernel_initializer=w_init,
                                        name='o1/decon2d'))
    layers.append(
        tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init,
                                           name='o1/batch_norm'))
    layers.append(tf.keras.layers.ReLU())
    # 32, 32 = s8

    layers.append(
        tf.keras.layers.Conv2DTranspose(filters=gf_dim * 2,
                                        kernel_size=(5, 5),
                                        strides=(2, 2),
                                        padding='SAME',
                                        kernel_initializer=w_init,
                                        name='o2/decon2d'))
    layers.append(
        tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init,
                                           name='o2/batch_norm'))
    layers.append(tf.keras.layers.ReLU())
    # 64, 64 = s4

    layers.append(
        tf.keras.layers.Conv2DTranspose(filters=gf_dim,
                                        kernel_size=(5, 5),
                                        strides=(2, 2),
                                        padding='SAME',
                                        kernel_initializer=w_init,
                                        name='o3/decon2d'))
    layers.append(
        tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init,
                                           name='o3/batch_norm'))
    layers.append(tf.keras.layers.ReLU())
    # 128, 128 = s2

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
