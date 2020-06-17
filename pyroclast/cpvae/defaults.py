"""
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
"""


def mnist():
    return dict(encoder='mnist_encoder',
                decoder='mnist_decoder',
                layers=1,
                optimizer='adam',
                learning_rate=3e-4,
                batch_size=64,
                embedding_dim=32,
                num_embeddings=512,
                commitment_cost=0.1,
                max_epochs=100,
                patience=20,
                class_loss_coeff=0.03)
    #model_name='vqvae_mnist')


def cifar10():
    return dict(encoder='vqvae_cifar10_encoder',
                decoder='vqvae_cifar10_decoder',
                layers=2,
                optimizer='adam',
                learning_rate=3e-4,
                batch_size=128,
                embedding_dim=64,
                num_embeddings=1024,
                commitment_cost=0.25,
                max_epochs=400,
                patience=80,
                model_name='vqvae_cifar10')
