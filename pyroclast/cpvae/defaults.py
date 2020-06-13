def mnist():
    return dict(
        encoder='mnist_encoder',
        decoder='mnist_decoder',
        latent_dim=98,
        max_epochs=1000,
        optimizer='adam',
        learning_rate=3e-4,
        prior='iso_gaussian_prior',
        posterior='diag_gaussian_posterior',
        output_distribution=
        'disc_logistic_posterior',  # disc_logistic or l2 or bernoulli
        beta=1.,
        patience=12,
        batch_size=32,
        model_name='vae_mnist')


def cifar10():
    return dict(encoder='vqvae_cifar10_encoder',
                decoder='vqvae_cifar10_decoder',
                optimizer='adam',
                learning_rate=3e-4,
                batch_size=128,
                max_epochs=400,
                patience=80,
                model_name='vqvae_cifar10')
