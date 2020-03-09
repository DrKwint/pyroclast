import os.path as osp

import tensorflow as tf
from absl.testing import parameterized

from pyroclast.common.feature_classifier import FeatureClassifierMixin
from pyroclast.common.tf_util import setup_tfds


class BasicMnistFeatureClassifier(FeatureClassifierMixin, tf.Module):

    def __init__(self):
        super(BasicMnistFeatureClassifier, self).__init__()
        self.conv_layer_1 = tf.keras.layers.Conv2D(16,
                                                   3,
                                                   padding='same',
                                                   activation=tf.nn.relu)
        self.conv_layer_2 = tf.keras.layers.Conv2D(
            16,
            3,
            padding='same',
            bias_initializer='glorot_normal',
            activation=tf.nn.relu)
        self.flatten_layer = tf.keras.layers.Flatten()
        self.first_dense = tf.keras.layers.Dense(25, activation=tf.nn.relu)
        self.num_features = 25
        self.dense_layer = tf.keras.layers.Dense(10)

    def __call__(self, x):
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

    def get_classification_module(self):
        return self.dense_layer


class FeatureClassifierMixinTest(parameterized.TestCase):

    def setUp(self):
        super(FeatureClassifierMixinTest, self).__init__()
        self.args = dict()
        self.args['data_limit'] = 24
        self.args['batch_size'] = 8
        self.args[
            'model_params_prefix'] = './feature_classifier_test/mnist_params'

        self.model = BasicMnistFeatureClassifier()
        self.ds = setup_tfds('mnist',
                             self.args['batch_size'],
                             None,
                             self.args['data_limit'],
                             shuffle_seed=431)

        # if saved model isn't present, train and save, otherwise load
        checkpoint = tf.train.Checkpoint(model=self.model)
        if not osp.exists(self.args['model_params_prefix'] + '-1.index'):
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
            for batch in self.ds['train']:
                with tf.GradientTape() as tape:
                    mean_loss = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=batch['label'],
                            logits=self.model(
                                tf.cast(batch['image'], tf.float32))))
                gradients = tape.gradient(mean_loss,
                                          self.model.trainable_variables)

                optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables))
            checkpoint.save(self.args['model_params_prefix'])
        else:
            for batch in self.ds['train']:
                self.model(tf.cast(batch['image'], tf.float32))
                break
            checkpoint.restore(self.args['model_params_prefix'] +
                               '-1').assert_consumed()

    def test_basic_mnist_feature_classifier(self):
        for batch in self.ds['train']:
            x = tf.cast(batch['image'], tf.float32)
            y = self.model(x)
            assert y.shape == (self.args['batch_size'], 10)

            f = self.model.features(x)
            assert f.shape == (self.args['batch_size'], self.model.num_features)

    def test_usefulness_and_robustness(self):
        # usefulness
        usefulness = self.model.usefulness(
            self.ds['train'].map(lambda x:
                                 (tf.cast(x['image'], tf.float32), x['label'])),
            self.ds['num_classes'])
        assert usefulness.shape == [self.model.num_features,
                                    10]  # has expected shape
        assert tf.math.reduce_any(tf.cast(usefulness, tf.bool))  # not all 0
        assert tf.math.reduce_max(
            usefulness) <= 1. + 1e-5  # has expected max value
        assert tf.math.reduce_min(
            usefulness) >= -1. - 1e-5  # has expected max value
        assert not tf.math.reduce_any(tf.math.is_nan(usefulness))  # not nan

        # robustness
        robustness = self.model.robustness(
            self.ds['train'].map(
                lambda x: (tf.cast(x['image'], tf.float32), x['label'])), 0, 0,
            self.ds['num_classes'], 0.1, 2)
        assert usefulness.shape == robustness.shape
        mean_usefulness = tf.reduce_mean(usefulness)
        mean_robustness = tf.reduce_mean(robustness)
        assert mean_robustness != mean_usefulness
        assert tf.math.reduce_max(
            robustness) <= 1. + 1e-5  # has expected max value
        assert tf.math.reduce_min(
            robustness) >= -1. - 1e-5  # has expected max value
        assert not tf.math.reduce_any(tf.math.is_nan(robustness))  # not nan
