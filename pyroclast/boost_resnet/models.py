import tensorflow as tf


def repr_module(channels):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            channels, 3, padding='same', activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            channels, 3, padding='same', activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization()
    ])


def classification_module(num_classes):
    if num_classes == 1:
        activation = tf.nn.tanh
    else:
        activation = None
    return tf.keras.layers.Dense(
        num_classes,
        kernel_initializer=tf.keras.initializers.RandomNormal(),
        activation=activation)
