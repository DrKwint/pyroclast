import pyroclast.classification.semisupervised.models as models
import pyroclast.classification.semisupervised.m2vae as m2vae

import tensorflow as tf
import tensorflow_probability as tfp


def learn(data, classifier_network, encoder_network, decoder_network, seed,
          total_epochs, classifier, encoder, decoder, latent_shape, prior,
          posterior, output_dist):
    classifier = models.build_classifier(classifier_network)
    encoder = models.build_encoder(encoder_network)
    decoder = models.build_decoder(decoder_network)

    prior = tfp.distributions.Normal(
        tf.zeros(latent_shape), tf.ones(latent_shape))
    # TODO: posterior and output distributions
    posterior = None
    output_dist = None

    model = m2vae.M2VAE(
        classifier=classifier,
        encoder=encoder,
        decoder=decoder,
        prior=prior,
        posterior=posterior,
        output_dist=output_dist,
    )

    # TODO: training loop
