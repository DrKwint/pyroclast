from absl.testing import parameterized
from pyroclast.common.tf_util import setup_tfds
from pyroclast.cpvae.learn import learn


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

    def test_mnist(self):
        # takes ~4 seconds on a laptop
        mnist_ds = setup_tfds(self.args, data_limit=10)
        learn(mnist_ds,
              encoder=self.args['encoder'],
              decoder=self.args['decoder'],
              latent_dim=self.args['latent_dim'],
              epochs=self.args['epochs'])
