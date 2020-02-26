import abc
import numpy as np
import tensorflow as tf


class VisualizableMixin(abc.ABC):
    """Interface Model interface.

    Provides functionality for various calculations and
    visualizations. Should be used as a mixin in conjunction with
    tf.Module.
    """

    @abc.abstractmethod
    def classify(self, x):
        """Calculates output logits given an input tensor.

        Args:
           x (tf.Tensor): Input of shape [batch_size, ...data_shape]

        Returns:
            logits (tf.Tensor): shape [batch_size, num_classes]
        """
        pass

    @property
    @abc.abstractmethod
    def conv_stack_submodel(self):
        """Returns an iterable of layers in the convolutional stack of a model

        This method is used to retrieve the conv stack of a network in
        order to find activation maps.

        Returns:
           conv_layers (iterable of tf.Module): An iterable of layers in the convolution stack
        """
        pass

    def activation_map(self, x, layer_index=-1):
        """Calculates the activation map of a layer of the convolution stack

        Calculates a basic activation map of the neurons in a layer of
        the convolution stack. This works by simply passing the data
        through the convlution stack of the model and then upscaling
        the resulting output of each filter to the width and height of
        the original data.

        Args:
           x (np.array): shape [batch_size, ...data_shape]
           layer_index (int): The index of the layer in the convolutional stack

        Returns:
           activation_map (np.array): shape [batch_size, ...data_shape]
        """
        if layer_index < 0:
            layer_index += len(self.conv_stack_submodel)
        layer_index = max(0, layer_index)

        result = x
        for i, layer in enumerate(self.conv_stack_submodel):
            result = layer(result)
            if i == layer_index:
                break
        return tf.image.resize(result, [x.shape[1], x.shape[2]])

    def cam_map(self, x):
        """Calculates the class activation mapping
        https://http://cnnlocalization.csail.mit.edu/

        Args: x (np.array): shape [batch_size, ...data_shape]

        Returns:
           activation_map (np.array): shape [batch_size, ...data_shape]

        TODO: This
        """
        pass

    def sensitivity_map(self, x, softmax=False):
        """Calculates the sensitivity map by back propagating on the input
        data x.

        Given input (images) x, calculates the sensitivity map of the
        input by first doing a forward pass and then backpropagating
        to find the gradients on the input.

        Args:
           x (np.array): shape [batch_size, ...data_shape]
           softmax (bool): Whether to softmax before calculating gradients

        Returns:
           sensitivity_map (np.array): shape [batch_size, ...data_shape]
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            result = self.classify(x)
            if softmax:
                result = tf.nn.softmax(result)
            grads = tape.gradient(result, x)
        return grads

    def smooth_grad(self, x, n=50, sigma=0.01):
        """Computes the smooth grad saliency-map

        SmoothGrad works by averaging the gradient of n input images
        that have been perturbed with Gausssian noise. Gaussian noise
        is parameterized by the standard deviation (sigma).

        Args:
           x (np.array): shape [batch_size, ...data_shape]
           n (int): The number of samples to collect
           sigma (float): The standard deviation of the Gaussian noise

        Returns:
           sensitivity_map (np.array): shape [batch_size, ...data_shape]

        """
        tile_scheme = [n] + [1 for _ in enumerate(x.shape)]
        x_dist = tf.tile(tf.constant(x)[None, :], tile_scheme)
        x_dist += tf.random.normal(x_dist.shape, stddev=sigma)
        x_dist = tf.reshape(x_dist, [-1, *x.shape[1:]])

        with tf.GradientTape() as tape:
            tape.watch(x_dist)
            result = self.classify(x_dist)
            gradients = tape.gradient(result, x_dist)

        gradients = tf.reshape(gradients, [n] + x.shape)
        print(x.shape, gradients.shape)
        gradients = tf.math.reduce_sum(gradients, axis=0)
        gradients /= n
        print(x.shape, gradients.shape)
        assert (gradients.shape == x.shape)
        return gradients
