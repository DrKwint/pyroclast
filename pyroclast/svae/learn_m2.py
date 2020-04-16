from pyroclast.svae.models import build_classifier, build_encoder, build_decoder
from pyroclast.svae.m2vae import M2VAE

import tensorflow as tf
import tensorflow_probability as tfp


def learn(train_data,
          seed,
          latent_dim,
          num_classes,
          classifier_network='conv_only',
          encoder_network='conv_only',
          decoder_network='upscale_conv',
          latent_prior='std_normal',
          latent_posterior='softplus_normal',
          output_dist='bernoulli'):
    """TODO: doc"""
    classifier = build_classifier(classifier_network, num_classes)
    encoder = build_encoder(encoder_network, latent_dim)
    decoder = build_decoder(decoder_network)

    # TODO: make these pull from a distributions file in pyroclast.common
    latent_prior = tfp.distributions.Normal(tf.zeros(latent_dim),
                                            tf.ones(latent_dim))
    latent_posterior = lambda loc, scale: tfp.distributions.Normal(
        loc, tf.nn.softplus(scale))
    class_prior = tfp.distributions.Bernoulli(logits=tf.ones(num_classes))
    class_posterior = lambda logits: tfp.distributions.Independent(
        tfp.distributions.Bernoulli(logits=logits))
    output_dist = lambda loc: tfp.distributions.Bernoulli(loc)

    model = M2VAE(classifier=classifier,
                  encoder=encoder,
                  decoder=decoder,
                  latent_prior=latent_prior,
                  latent_posterior=latent_posterior,
                  class_prior=class_prior,
                  class_posterior=class_posterior,
                  output_dist=output_dist,
                  num_classes=num_classes)

    # build graph
    data_labeled, data_unlabeled = train_data
    data_labeled['image'] = tf.cast(data_labeled['image'], tf.float32)
    data_labeled['label'] = tf.one_hot(data_labeled['label'], num_classes)
    data_unlabeled['image'] = tf.cast(data_unlabeled['image'], tf.float32)

    # loss calculation
    loss = model.loss(data_labeled['image'], data_labeled['label'],
                      data_unlabeled['image'])

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    # training loop
    with tf.Session() as session:
        session.run(tf.variables_initializer(tf.global_variables()))
        i = 0
        while True:
            print("step {}".format(i))
            i += 1
            session.run(train_op)

    return model
