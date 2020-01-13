import abc
import numpy as np
import tensorflow as tf


class InterpretableModel(abc.ABC):
    """Interface Model interface.

    Provides functionality for various calculations and
    visualizations. Should be used as a mixin in conjunction with
    tf.Module.

    """

    @abc.abstractmethod
    def classify(self, x):
        """Calculates output logits given a set of inputs.

        Returns:
            logits (tf.Tensor): shape [batch_size, num_classes]
        """
        pass

    @abc.abstractmethod
    def conv_stack_submodel(self):
        """Returns the submodel that contains the convolution stack

        This method is used to retrieve the conv stack of a network in
        order to find activation maps.

        Returns:
           submodel (tf.Module): A submodel that begins at the first layer of the network and includes the conv stack.

        """
        pass

    def activation_map(self, x):
        """Calculates the activation map of the final layer of a convolution stack

        Calculates a basic activation map of the neurons in the final
        layer of a convolution stack. This works by simply passing the
        data through the convlution stack of the model and then
        upscaling the resulting output of each filter to the original
        data shape.

        Args:
           x (np.array): shape [batch_size, ...data_shape]

        Returns:
           activation_map (np.array): shape [batch_size, ...data_shape]

        """
        result = self.conv_stack_submodel(x)
        return tf.image.resize_images(result, [x.shape[1], x.shape[2]])

    def cam_map(self, x):
        """Calculates the class activation mapping
        https://http://cnnlocalization.csail.mit.edu/

        Args: x (np.array): shape [batch_size, ...data_shape]

        Returns:
           activation_map (np.array): shape [batch_size, ...data_shape]

        TODO: This
        """
        pass

    def sensitivity_map(self, x, y, softmax=False):
        """Calculates the sensitivity map by back propagating on the input
        data x.

        Given input (images) x, calculates the sensitivity map by
        first doing a forward pass and then backpropagating to find
        the gradients on the input.

        Args:
           x (np.array): shape [batch_size, ...data_shape]

        Returns:
           sensitivity_map (np.array): shape [batch_size, ...data_shape]

        """
        with tf.GradientTape() as tape:
            result = self.classify(x)
            if softmax:
                result = tf.nn.softmax(result)
            grads = tape.gradient(result[:, y], x)
        return grads
