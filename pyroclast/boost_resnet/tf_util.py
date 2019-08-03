import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.compat.v1.layers.batch_normalization(
        inputs=inputs,
        axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
        center=True,
        scale=True,
        training=training,
        fused=True)


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(
            tensor=inputs,
            paddings=[[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(
            tensor=inputs,
            paddings=[[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.compat.v1.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False,
        kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
        data_format=data_format)


def _building_block_v2(inputs, filters, training, projection_shortcut, strides,
                       data_format):
    """A single block for ResNet v2, without a bottleneck.
  Batch normalization then ReLu then convolution as described by:
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/pdf/1603.05027.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block; shape should match inputs.
  """
    shortcut = inputs
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=strides,
        data_format=data_format)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=1,
        data_format=data_format)

    return inputs + shortcut
