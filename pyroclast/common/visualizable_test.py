import tensorflow as tf
from absl.testing import parameterized

from pyroclast.common.tf_util import setup_tfds
from pyroclast.common.visualizable import VisualizableMixin


class BasicMnistVisualizable(VisualizableMixin, tf.Module):

    def __init__(self):
        super(BasicMnistVisualizable, self).__init__()
        self.conv_layer = tf.keras.layers.Conv2D(16, 3, padding='same')
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer = tf.keras.layers.Dense(10)

    def __call__(self, x, y=None):
        x = self.conv_layer(x)
        x = self.flatten_layer(x)
        x = self.dense_layer(x)
        return x

    def classify(self, x):
        return self(x)

    def conv_stack_submodel(self):
        return self.conv_layer


class VisualizableMixinTest(parameterized.TestCase):

    def setUp(self):
        super(VisualizableMixinTest, self).__init__()
        self.args = dict()
        self.args['data_limit'] = 80
        self.args['batch_size'] = 8

        self.model = BasicMnistVisualizable()
        self.ds = setup_tfds('mnist', self.args['batch_size'],
                             self.args['data_limit'])

    def test_basic_mnist_visualizable(self):
        for batch in self.ds['train']:
            x = tf.cast(batch['image'], tf.float32)
            y = self.model(x)
            assert y.shape == (self.args['batch_size'], 10)

    def test_activation_map(self):
        for batch in self.ds['train']:
            x = tf.cast(batch['image'], tf.float32)
            activation_maps = self.model.activation_map(x)
            assert activation_maps.shape[0:3] == x.shape[0:3]

    def test_cam_map(self):
        pass

    def test_sensitivity_map(self):
        for batch in self.ds['train']:
            x = tf.cast(batch['image'], tf.float32)
            y = self.model(x)
            input_sensitivity_maps = self.model.sensitivity_map(x, 0)
            assert input_sensitivity_maps is not None
            assert input_sensitivity_maps.shape[0:3] == x.shape[0:3]
