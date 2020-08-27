import tensorflow as tf
import sonnet as snt


class WeightNorm(tf.Module):

    def __init__(self, layer, data_init=True):
        self._layer = layer
        self._data_init = data_init
        self._filter_axis = -1  # this is the case on Linear and Conv2D layers, but maybe not others

    @tf.function
    def __call__(self, inputs):
        self._initialize(inputs)
        if not self._initialized:
            if self._data_init:
                self._data_dep_init(inputs)
            else:
                # initialize `g` as the norm of the initialized kernel
                self._init_norm()
            self._initialized.assign(True)
        self._compute_weights()
        output = self._layer(inputs)
        return output

    def _compute_weights(self):
        new_axis = 0
        self._layer.w = tf.nn.l2_normalize(
            self.v, axis=self._kernel_norm_axes) * tf.expand_dims(
                self.g, new_axis)

    @snt.once
    def _initialize(self, inputs):
        self._layer._initialize(inputs)
        if not hasattr(self._layer, 'w'):
            raise ValueError('`WeightNorm` must wrap a layer that'
                             ' contains a `w` for weights')

        self._kernel_norm_axes = list(range(self._layer.w.shape.ndims))
        self._kernel_norm_axes.pop(self._filter_axis)
        self.v = self._layer.w

        # to avoid a duplicate `w` variable after `build` is called
        self.g = tf.Variable(tf.ones(int(self.v.shape[-1]), dtype=self.v.dtype),
                             name='g')

        self._initialized = tf.Variable(False,
                                        name='initialized',
                                        dtype=tf.bool,
                                        trainable=False)
        self._initialized.assign(False)

    def _init_norm(self):
        """Set the norm of the weight vector."""
        kernel_norm = tf.sqrt(
            tf.reduce_sum(tf.square(self.v), axis=self._kernel_norm_axes))
        self.g = kernel_norm

    def _data_dep_init(self, inputs):
        self._compute_weights()

        if self._layer.with_bias:
            bias = self._layer.b
            self._layer.bias = tf.zeros_like(bias)

        x_init = self._layer(inputs)
        norm_axes_out = list(range(x_init.shape.rank - 1))
        m_init, v_init = tf.nn.moments(x_init, norm_axes_out)
        scale_init = 1. / tf.sqrt(v_init + 1e-10)

        self.g.assign(self.g * scale_init)
        if self._layer.with_bias:
            self._layer.b = bias
            self._layer.bias = (-1 * m_init * scale_init)
