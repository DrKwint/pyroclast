import sklearn
import tensorflow as tf
from absl.testing import parameterized

from pyroclast.common.tf_util import setup_tfds
from pyroclast.cpvae.model import CpVAE
from pyroclast.cpvae.tf_models import VAEDecoder, VAEEncoder
from pyroclast.cpvae.ddt import DDT


class CpVAETest(parameterized.TestCase):

    def setUp(self):
        super(CpVAETest, self).setUp()
        self.args = dict()
        self.args['dataset'] = 'mnist'
        self.args['encoder'] = 'mnist_encoder'
        self.args['decoder'] = 'mnist_decoder'
        self.args['epochs'] = 1
        self.args['data_limit'] = 80
        self.args['latent_dim'] = 10
        self.args['batch_size'] = 8
        self.args['output_dir'] = 'cpvae_cpvae_test'

        self.ds = setup_tfds(self.args['dataset'], self.args['batch_size'],
                             None, self.args['data_limit'])
        self.encoder = VAEEncoder(self.args['encoder'], self.args['latent_dim'])
        self.decoder = VAEDecoder(self.args['decoder'], self.ds['shape'][-1])
        decision_tree = sklearn.tree.DecisionTreeClassifier(
            max_depth=2, min_weight_fraction_leaf=0.01, max_leaf_nodes=4)
        self.DDT = DDT(decision_tree, 10)

    def test_vae_loss(self):
        output_distributions = ['disc_logistic', 'l2', 'bernoulli']
        for dist in output_distributions:
            # require the outputs of CpVAE loss fn to be non-negative and have size equal to batch size
            model = CpVAE(self.encoder,
                          self.decoder,
                          self.DDT,
                          self.args['latent_dim'],
                          self.ds['num_classes'],
                          1,
                          output_dist=dist)
            model.classifier.update_model_tree(self.ds['train'],
                                               model.posterior)

            for batch in self.ds['train']:
                x = tf.cast(batch['image'], tf.float32)
                x_hat, _, z_posterior, x_hat_scale = model(x)
                distortion, rate = model.vae_loss(x, x_hat, x_hat_scale,
                                                  z_posterior)
                assert distortion.shape == self.args[
                    'batch_size'], 'output_dist is {}'.format(dist)
                # assert not any(distortion < 0.)
                assert rate.shape == self.args['batch_size']
                assert not any(rate < 0.)
                break
