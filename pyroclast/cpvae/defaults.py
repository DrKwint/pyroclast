def mnist():
    return dict(
        encoder='mnist_encoder',
        decoder='mnist_decoder',
        latent_dim=64,
        epochs=1000,
        oversample=10,
        max_tree_depth=5,
        tree_update_period=3,
        optimizer='rmsprop',  # adam or rmsprop
        learning_rate=3e-4,
        prior='iso_gaussian_prior',
        posterior='diag_gaussian_posterior',
        output_distribution=
        'disc_logistic_posterior',  # disc_logistic or l2 or bernoulli
        num_samples=5,
        clip_norm=0.,
        alpha=1.,
        beta=1.,
        gamma=1.,
        omega=1.,
        patience=12)
