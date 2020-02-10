import tensorflow as tf

from pyroclast.common.models import register


@register('vgg19_conv')
def vgg19_conv():
    return tf.keras.applications.VGG19(include_top=False, pooling=None)


@register('mnist_conv')
def mnist_conv():
    output_channels = [32, 64, 128]
    kernel_shapes = [5, 3, 3]
    strides = [2, 2, 1]
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(oc,
                               ks,
                               strides,
                               activation=tf.nn.relu,
                               padding='same')
        for (oc, ks, strides) in zip(output_channels, kernel_shapes, strides)
    ])
