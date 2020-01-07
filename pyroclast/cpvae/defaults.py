
def mnist():
    return dict(
        encoder='mnist_encoder',
        decoder='mnist_decoder',
        latent_dim=32,
        epochs=100,
        alpha=1.,
        beta=1.,
        gamma=10.)