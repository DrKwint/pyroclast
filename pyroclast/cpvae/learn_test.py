from absl.testing import parameterized
from pyroclast.common.tf_util import setup_tfds
from pyroclast.cpvae.cpvae import learn


class LearnTest(parameterized.TestCase):

    def setUp(self):
        super(LearnTest, self).setUp()
        self.args = dict()
        self.args['dataset'] = 'mnist'
        self.args['encoder'] = 'mnist_encoder'
        self.args['decoder'] = 'mnist_decoder'
        self.args['epochs'] = 1
        self.args['latent_dim'] = 10
        self.args['batch_size'] = 8
        self.args['data_limit'] = 80
        self.args['output_dir'] = 'cpvae_learn_test'

    def test_mnist(self):
        # takes ~4 seconds on a laptop
        mnist_ds = setup_tfds(self.args['dataset'], self.args['batch_size'], self.args['data_limit'])
        learn(mnist_ds,
              encoder=self.args['encoder'],
              decoder=self.args['decoder'],
              latent_dim=self.args['latent_dim'],
              epochs=self.args['epochs'],
              output_dir=self.args['output_dir'])
