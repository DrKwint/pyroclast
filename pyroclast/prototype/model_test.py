import tensorflow as tf
from absl.testing import parameterized

from pyroclast.common.models import get_network_builder
from pyroclast.common.tf_util import setup_tfds
from pyroclast.prototype.model import ProtoPNet


class PrototypeModelTest(parameterized.TestCase):

    def setUp(self):
        self.args = dict()
        self.args['batch_size'] = 8
        self.args['data_limit'] = 24
        self.args['num_prototypes'] = 100
        self.args['prototype_dim'] = 64
        self.ds = setup_tfds('mnist', self.args['batch_size'], None,
                             self.args['data_limit'])
        conv_stack = get_network_builder('mnist_conv')()
        self.model = ProtoPNet(conv_stack, self.args['num_prototypes'],
                               self.args['prototype_dim'],
                               self.ds['num_classes'])

    def test_basic_functionality(self):
        for batch in self.ds['train']:
            features = self.model.features(tf.cast(batch['image'], tf.float32))
            classification = self.model.classify_features(features)
            # check that the features shape is what the mixin docs expect
            assert features.shape == (self.args['batch_size'],
                                      self.args['num_prototypes'])
            assert classification.shape == (self.args['batch_size'],
                                            self.ds['num_classes'])

    def test_usefulness_robustness(self):
        usefulness = self.model.usefulness(
            self.ds['test'].map(lambda x:
                                (tf.cast(x['image'], tf.float32), x['label'])),
            self.ds['num_classes'])
        assert usefulness.shape == (self.args['num_prototypes'],
                                    self.ds['num_classes'])

        robustness = self.model.robustness(self.ds['test'].map(
            lambda x: (tf.cast(x['image'], tf.float32), x['label'])),
                                           self.ds['num_classes'],
                                           eps=1.,
                                           norm=2)
        assert robustness.shape == (self.args['num_prototypes'],
                                    self.ds['num_classes'])
