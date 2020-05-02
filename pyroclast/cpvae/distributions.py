import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from collections import Iterable

tfd = tfp.distributions
tfb = tfp.bijectors

mapping = {}


def register(name):

    def _thunk(func):
        mapping[name] = func
        return func

    return _thunk


@register("iso_gaussian_prior")
def iso_gaussian_prior(latent_dimension):
    dist = tfd.MultivariateNormalDiag(loc=tf.zeros(latent_dimension),
                                      scale_diag=tf.ones(latent_dimension))
    if isinstance(latent_dimension, Iterable):
        dist = tfd.Independent(dist, len(latent_dimension) - 1)
    return dist


@register("iaf_prior")
def iaf_prior(latent_dimension, ar_network):
    return tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=0., scale=1.),
        bijector=tfb.Invert(
            tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=ar_network)),
        event_shape=[latent_dimension])


@register("iaf_posterior")
def iaf_posterior():
    return lambda loc, scale, ar_network: tfd.TransformedDistribution(
        distribution=tfd.MultivariateNormalDiag(loc, scale),
        bijector=tfb.Invert(
            tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=ar_network)))


@register("diag_gaussian_posterior")
def diag_gaussian_posterior():
    return lambda loc, scale_diag: tfp.distributions.MultivariateNormalDiag(
        loc=loc, scale_diag=scale_diag)


@register("disc_logistic_posterior")
def disc_logistic_posterior():
    return lambda loc, scale: tfp.distributions.Independent(
        tfd.QuantizedDistribution(distribution=tfd.TransformedDistribution(
            distribution=tfd.Logistic(loc, scale),
            bijector=tfb.Shift(-0.5 / 256)),
                                  low=0.,
                                  high=1.), 3)


LEARNED_GAUSSIAN_CLASS_PRIOR_FN = lambda latent_dimension, class_num: [
    tfp.distributions.MultivariateNormalDiag(
        loc=tf.Variable(np.zeros(latent_dimension, dtype=np.float32),
                        name='class_{}_loc'.format(i)),
        scale_diag=tfp.util.DeferredTensor(
            tf.Variable(np.ones(latent_dimension, dtype=np.float32),
                        name='class_{}_scale_diag'.format(i)), tf.math.softplus
        )) for i in range(class_num)
]


@register("mnist_pixelcnn_posterior")
def mnist_pixelcnn_posterior():
    return tfd.PixelCNN(image_shape=(28, 28, 1),
                        conditional_shape=(28, 28, 1),
                        num_resnet=2,
                        num_hierarchies=2,
                        num_filters=32,
                        num_logistic_mix=4,
                        dropout_p=.3,
                        low=0.,
                        high=1.)


MADE = lambda latent_dimension: tfb.AutoregressiveNetwork(params=2,
                                                          event_shape=
                                                          latent_dimension,
                                                          hidden_units=[20, 20],
                                                          activation=tf.nn.relu,
                                                          input_order='random')


def get_distribution_builder(name):
    """
    If you want to register your own network outside models.py, you just need:

    Usage Examplee:
    -------------
    from pyroclast.common.models import register
    @register("your_network_name")
    def your_network_define(**net_kwargs):
        ...
        return network_fn

    """
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown network type: {}'.format(name))
