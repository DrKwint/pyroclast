import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

GAUSSIAN_PRIOR_FN = lambda latent_dimension: tfp.distributions.MultivariateNormalDiag(
    loc=tf.zeros(latent_dimension), scale_diag=tf.ones(latent_dimension))

GAUSSIAN_POSTERIOR_FN = lambda loc, scale_diag: tfp.distributions.MultivariateNormalDiag(
    loc=loc, scale_diag=scale_diag)

LEARNED_GAUSSIAN_CLASS_PRIOR_FN = lambda latent_dimension, class_num: [
    tfp.distributions.MultivariateNormalDiag(
        loc=tf.Variable(np.zeros(latent_dimension, dtype=np.float32),
                        name='class_{}_loc'.format(i)),
        scale_diag=tfp.util.DeferredTensor(
            tf.Variable(np.ones(latent_dimension, dtype=np.float32),
                        name='class_{}_scale_diag'.format(i)), tf.math.softplus
        )) for i in range(class_num)
]

MNIST_PIXELCNN = tfd.PixelCNN(
    image_shape=(28, 28, 1),
    num_resnet=1,
    num_hierarchies=2,
    num_filters=32,
    num_logistic_mix=5,
    dropout_p=.3,
)

DISCRETIZED_LOGISTIC_FN = lambda loc, scale: tfp.distributions.Independent(
    tfd.QuantizedDistribution(distribution=tfd.TransformedDistribution(
        distribution=tfd.Logistic(loc, scale), bijector=tfb.Shift(-0.5 / 256)),
                              low=0.,
                              high=1.), 3)

MADE = lambda latent_dimension: tfb.AutoregressiveNetwork(params=2,
                                                          event_shape=
                                                          latent_dimension,
                                                          hidden_units=[20, 20],
                                                          activation=tf.nn.relu,
                                                          input_order='random')
