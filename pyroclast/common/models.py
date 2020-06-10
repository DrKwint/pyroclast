import tensorflow as tf
import sonnet as snt

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
        tf.keras.layers.Conv2D(32,
                               3,
                               2,
                               padding='SAME',
                               activation=tf.nn.leaky_relu))  # 14 x 14 x 32
    layers.append(
        tf.keras.layers.Conv2D(32,
                               3,
                               padding='SAME',
                               activation=tf.nn.leaky_relu))
    layers.append(
        tf.keras.layers.Conv2D(64,
                               3,
                               2,
                               padding='SAME',
                               activation=tf.nn.leaky_relu))  # 7 x 7 x 64
    layers.append(
        tf.keras.layers.Conv2D(64,
                               3,
                               padding='SAME',
                               activation=tf.nn.leaky_relu))
    return tf.keras.Sequential(layers)


@register("mnist_decoder")
def mnist_deconv():
    # expects inputs of shape [N, 98]
    layers = []
    layers.append(tf.keras.layers.Reshape([7, 7, 2]))
    layers.append(
        tf.keras.layers.Conv2DTranspose(filters=64,
                                        kernel_size=5,
                                        strides=2,
                                        padding='SAME',
                                        activation=tf.nn.leaky_relu))
    layers.append(
        tf.keras.layers.Conv2D(filters=64,
                               kernel_size=3,
                               padding='SAME',
                               activation=tf.nn.leaky_relu))
    layers.append(
        tf.keras.layers.Conv2DTranspose(filters=32,
                                        kernel_size=5,
                                        strides=2,
                                        padding='SAME',
                                        activation=tf.nn.leaky_relu))
    layers.append(
        tf.keras.layers.Conv2D(filters=32,
                               kernel_size=3,
                               padding='SAME',
                               activation=tf.nn.leaky_relu))
    return tf.keras.Sequential(layers)


@register("cifar10_encoder")
def cifar10_conv():
    layers = []

    layers.append(  #16x16
        tf.keras.layers.Conv2D(32, (3, 3), (2, 2), padding='SAME'))
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.LeakyReLU())
    layers.append(tf.keras.layers.Conv2D(32, (3, 3), (1, 1), padding='SAME'))
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.LeakyReLU())

    layers.append(  #8x8
        tf.keras.layers.Conv2D(64, (3, 3), (2, 2), padding='SAME'))
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.LeakyReLU())
    layers.append(tf.keras.layers.Conv2D(64, (3, 3), (1, 1), padding='SAME'))
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.LeakyReLU())
    layers.append(  #4x4
        tf.keras.layers.Conv2D(64, (3, 3), (2, 2), padding='SAME'))
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.LeakyReLU())
    layers.append(tf.keras.layers.Conv2D(64, (3, 3), (1, 1), padding='SAME'))
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.LeakyReLU())
    layers.append(  #2x2
        tf.keras.layers.Conv2D(128, (3, 3), (2, 2), padding='SAME'))
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.LeakyReLU())
    layers.append(tf.keras.layers.Conv2D(128, (3, 3), (1, 1), padding='SAME'))
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.LeakyReLU())
    layers.append(  #1x1
        tf.keras.layers.Conv2D(256, (3, 3), (2, 2), padding='SAME'))
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.LeakyReLU())
    layers.append(tf.keras.layers.Conv2D(256, (3, 3), (1, 1), padding='SAME'))
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.LeakyReLU())
    return tf.keras.Sequential(layers)


@register('cifar10_decoder')
def cifar10_decoder():
    layers = []
    layers.append(tf.keras.layers.Reshape(target_shape=[1, 1, 256]))
    layers.append(  # 2x2
        tf.keras.layers.Conv2DTranspose(filters=128,
                                        kernel_size=3,
                                        strides=2,
                                        padding='SAME'))
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.LeakyReLU())
    layers.append(
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME'))
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.LeakyReLU())
    layers.append(  # 4x4
        tf.keras.layers.Conv2DTranspose(filters=64,
                                        kernel_size=3,
                                        strides=2,
                                        padding='SAME'))
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.LeakyReLU())
    layers.append(
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='SAME'))
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.LeakyReLU())
    layers.append(  # 8x8
        tf.keras.layers.Conv2DTranspose(filters=32,
                                        kernel_size=(3, 3),
                                        strides=(2, 2),
                                        padding='SAME'))
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.LeakyReLU())
    layers.append(
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='SAME'))
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.LeakyReLU())
    layers.append(  # 16x16
        tf.keras.layers.Conv2DTranspose(filters=16,
                                        kernel_size=(3, 3),
                                        strides=(2, 2),
                                        padding='SAME'))
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.LeakyReLU())
    layers.append(
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='SAME'))
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.LeakyReLU())
    layers.append(  # 32x32
        tf.keras.layers.Conv2DTranspose(filters=8,
                                        kernel_size=(3, 3),
                                        strides=(2, 2),
                                        padding='SAME'))
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.LeakyReLU())
    layers.append(
        tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='SAME'))
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.LeakyReLU())
    return tf.keras.Sequential(layers)


@register('vgg19')
def vgg19(**kwargs):
    return tf.keras.applications.VGG19(include_top=False,
                                       weights='imagenet',
                                       input_shape=kwargs['shape'],
                                       pooling=None)


class ResidualStack(snt.Module):

    def __init__(self,
                 num_hiddens,
                 num_residual_layers,
                 num_residual_hiddens,
                 name=None):
        super(ResidualStack, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._layers = []
        for i in range(num_residual_layers):
            conv3 = snt.Conv2D(output_channels=num_residual_hiddens,
                               kernel_shape=(3, 3),
                               stride=(1, 1),
                               name="res3x3_%d" % i)
            conv1 = snt.Conv2D(output_channels=num_hiddens,
                               kernel_shape=(1, 1),
                               stride=(1, 1),
                               name="res1x1_%d" % i)
            self._layers.append((conv3, conv1))

    def __call__(self, inputs):
        h = inputs
        for conv3, conv1 in self._layers:
            conv3_out = conv3(tf.nn.relu(h))
            conv1_out = conv1(tf.nn.relu(conv3_out))
            h += conv1_out
        return tf.nn.relu(h)  # Resnet V1 style


class Encoder(snt.Module):

    def __init__(self,
                 num_hiddens,
                 num_residual_layers,
                 num_residual_hiddens,
                 downscale=4,
                 name=None):
        super(Encoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._enc_1 = snt.Conv2D(output_channels=self._num_hiddens // 2,
                                 kernel_shape=(4, 4),
                                 stride=(2, 2),
                                 name="enc_1")
        if downscale == 4:
            self._enc_2 = snt.Conv2D(output_channels=self._num_hiddens,
                                     kernel_shape=(4, 4),
                                     stride=(2, 2),
                                     name="enc_2")
        self._enc_3 = snt.Conv2D(output_channels=self._num_hiddens,
                                 kernel_shape=(3, 3),
                                 stride=(1, 1),
                                 name="enc_3")
        self._residual_stack = ResidualStack(self._num_hiddens,
                                             self._num_residual_layers,
                                             self._num_residual_hiddens)

    def __call__(self, x):
        h = tf.nn.relu(self._enc_1(x))
        h = tf.nn.relu(self._enc_2(h))
        h = tf.nn.relu(self._enc_3(h))
        return self._residual_stack(h)


@register('vqvae_cifar10_encoder')
def vqvae_cifar10_enc():
    return Encoder(128, 3, 64)


@register('vqvae_cifar10_decoder')
def vqvae_cifar10_dec():
    return Decoder(128, 3, 64)


class Decoder(snt.Module):

    def __init__(self,
                 num_hiddens,
                 num_residual_layers,
                 num_residual_hiddens,
                 upscale=4,
                 name=None):
        super(Decoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._dec_1 = snt.Conv2D(output_channels=self._num_hiddens,
                                 kernel_shape=(3, 3),
                                 stride=(1, 1),
                                 name="dec_1")
        self._residual_stack = ResidualStack(self._num_hiddens,
                                             self._num_residual_layers,
                                             self._num_residual_hiddens)
        if upscale == 4:
            self._dec_2 = snt.Conv2DTranspose(
                output_channels=self._num_hiddens // 2,
                output_shape=None,
                kernel_shape=(4, 4),
                stride=(2, 2),
                name="dec_2")
        self._dec_3 = snt.Conv2DTranspose(output_channels=3,
                                          output_shape=None,
                                          kernel_shape=(4, 4),
                                          stride=(2, 2),
                                          name="dec_3")

    def __call__(self, x):
        h = self._dec_1(x)
        h = self._residual_stack(h)
        h = tf.nn.relu(self._dec_2(h))
        x_recon = self._dec_3(h)
        return x_recon


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
