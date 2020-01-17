def caltech_birds2011():
    return dict(num_prototypes=200, delay_conv_stack_training=True)


def mnist():
    return dict(epochs_phase_1=5,
                epochs_phase_3=5,
                num_prototypes=100,
                conv_stack='mnist_conv')
