from pyroclast.svae.models import build_classifier, build_encoder, build_decoder
from pyroclast.svae.m2vae import M2VAE
from pyroclast.common.data_registry import get_dataset

import tensorflow as tf
import tensorflow_probability as tfp


def learn(train_data,
          seed,
          latent_shape,
          classifier_network='conv_only',
          encoder_network='cnn',
          decoder_network='up_cnn',
          prior='std_normal',
          posterior='softplus_normal',
          output_dist='bernoulli'):
    classifier = build_classifier(classifier_network)
    encoder = build_encoder(encoder_network)
    decoder = build_decoder(decoder_network)

    prior = tfp.distributions.Normal(
        tf.zeros(latent_shape), tf.ones(latent_shape))
    # TODO: posterior and output distributions
    posterior = None
    output_dist = None

    model = M2VAE(
        classifier=classifier,
        encoder=encoder,
        decoder=decoder,
        prior=prior,
        posterior=posterior,
        output_dist=output_dist,
    )

    # build graph
    data_labeled, data_unlabeled = train_data
    temperature_ph = tf.placeholder(tf.float32, shape=())
    p_x_l, p_y_l, p_z_l = model((data_labeled, temperature_ph))
    p_x_u, p_y_u, p_z_u = model((data_unlabeled, temperature_ph))

    # TODO: training loop
    unsupervised_loss = model.unsupervised_loss(data_unlabeled, p_x_u, p_y_u,
                                                p_z_u)

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(unsupervised_loss)
