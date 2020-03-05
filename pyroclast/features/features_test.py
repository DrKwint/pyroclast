import tensorflow as tf
from absl.testing import parameterized

from pyroclast.common.models import get_network_builder
from pyroclast.common.tf_util import setup_tfds
from pyroclast.features.features import build_savable_objects, run_minibatch


class PrototypeModelTest(parameterized.TestCase):

    def setUp(self):
        self.args = dict()
        self.args['batch_size'] = 8
        self.args['data_limit'] = 24
        self.args['conv_stack_name'] = 'ross_net'
        self.args['output_dir'] = './features_test'
        self.args['model_save_name'] = 'features_test'

        self.ds = setup_tfds('mnist', self.args['batch_size'], None,
                             self.args['data_limit'])
        objects = build_savable_objects(self.args['conv_stack_name'], self.ds,
                                        0.0001, self.args['output_dir'],
                                        self.args['model_save_name'])

        self.model = objects['model']
        self.optimizer = objects['optimizer']
        self.global_step = objects['global_step']
        self.checkpoint = objects['checkpoint']
        self.ckpt_manager = objects['ckpt_manager']

    def test_basic_functionality(self):
        for batch in self.ds['train']:
            features = self.model.features(tf.cast(batch['image'], tf.float32))
            classification = self.model.classify_features(features)
            # check that the features shape is what the mixin docs expect
            assert classification.shape == (self.args['batch_size'],
                                            self.ds['num_classes'])

    def test_run_minibatch(self):
        writer = tf.summary.create_file_writer(self.args['output_dir'])
        for batch in self.ds['train']:
            run_minibatch(self.model, self.optimizer, self.global_step, 0,
                          batch, self.ds['num_classes'], 0., 0., writer)
            run_minibatch(self.model, self.optimizer, self.global_step, 0,
                          batch, self.ds['num_classes'], 1., 0., writer)
            run_minibatch(self.model, self.optimizer, self.global_step, 0,
                          batch, self.ds['num_classes'], 1., 1., writer)
            break
