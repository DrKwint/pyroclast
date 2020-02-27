.. default-role:: any

.. _visualizable:


Visualizable
============

The :class:`~pyroclast.common.visualizable.VisualizableMixin` provides
basic visualizations for a deep learning model.


Mixin Requirements
------------------

- :meth:`~pyroclast.common.visualizable.VisualizableMixin.classify`
  Override to call network and return logits given input :code:`x`.
- :meth:`~pyroclast.common.visualizable.VisualizableMixin.conv_stack_submodel`
  Override to return the final layer in the convolutional stack of the model.

Mixin Functionality
-------------------

- :meth:`~pyroclast.common.visualizable.VisualizableMixin.activation_map`
  Takes in input data :code:`x` and returns the activation map of the
  last convolutional layer in the convolutional stack returned by
  :meth:`~pyroclast.common.visualizable.VisualizableMixin.conv_stack_submodel`.

- :meth:`~pyroclast.common.visualizable.VisualizableMixin.cam_map`
  Generates the class activation mapping
  http://cnnlocalization.csail.mit.edu/ given input :code:`x`. This
  method has yet to be implemented.

- :meth:`~pyroclast.common.visualizable.VisualizableMixin.sensitivity_map`
  Generates the sensitivity map of input :code:`x`. Optional
  softmaxing can be applied before completing the forward pass.

- :meth:`~pyroclast.common.visualizable.VisualizableMixin.smooth_grad`
  Generates the SmoothGRAD sensitivity map of input :code:`x`.
