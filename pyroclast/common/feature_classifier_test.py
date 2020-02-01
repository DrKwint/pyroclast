import tensorflow as tf
from absl.testing import parameterized

from pyroclast.common.tf_util import setup_tfds
from pyroclast.common.feature_classifier import FeatureClassifierMixin


class BasicMnistFeatureClassifier(FeatureClassifierMixin, tf.Module):

    def __init__(self):
        super(BasicMnistFeatureClassifier, self).__init__()
        self.conv_layer_1 = tf.keras.layers.Conv2D(16, 3, padding='same')
        self.conv_layer_2 = tf.keras.layers.Conv2D(
            16, 3, padding='same', bias_initializer='glorot_normal')
        self.flatten_layer = tf.keras.layers.Flatten()
        self.first_dense = tf.keras.layers.Dense(25)
        self.num_features = 25
        self.dense_layer = tf.keras.layers.Dense(10)

    def __call__(self, x, y=None):
        f = self.features(x)
        return self.classify_features(f)

    def features(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.flatten_layer(x)
        x = self.first_dense(x)
        return x

    def classify_features(self, f):
        return self.dense_layer(f)


class FeatureClassifierMixinTest(parameterized.TestCase):

    def setUp(self):
        super(FeatureClassifierMixinTest, self).__init__()
        self.args = dict()
        self.args['data_limit'] = 24
        self.args['batch_size'] = 8

        self.model = BasicMnistFeatureClassifier()
        self.ds = setup_tfds('mnist', self.args['batch_size'], None,
                             self.args['data_limit'])

    def test_basic_mnist_feature_classifier(self):
        for batch in self.ds['train']:
            x = tf.cast(batch['image'], tf.float32)
            y = self.model(x)
            assert y.shape == (self.args['batch_size'], 10)

            f = self.model.features(x)
            assert f.shape == (self.args['batch_size'], self.model.num_features)

    def test_usefulness_and_robustness(self):
        # usefulness
        usefulness = self.model.usefulness(self.ds['train'])
        assert usefulness.shape == [self.model.num_features, 10]
        assert tf.math.reduce_any(tf.cast(usefulness, tf.bool))

        # robustness
        robustness = self.model.robustness(self.ds['train'], 0.1, 2)
        mean_usefulness = tf.reduce_mean(usefulness)
        mean_robustness = tf.reduce_mean(robustness)
        assert mean_robustness < mean_usefulness
