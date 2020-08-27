def learn_mnist():
    return dict(
        channels=16,
        num_embeddings=32,
        vq_commitment_cost=0.25,
        batch_size=64,
        learning_rate=3e-4,
        #output_dir='vqvae_mnist',
        patience=20)


def learn_prior_mnist():
    return dict(
        batch_size=128,
        learning_rate=3e-4,
        num_embeddings=32,
        #output_dir='vqvae_mnist',
        patience=20)
