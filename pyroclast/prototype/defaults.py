def caltech_birds2011():
    return dict(num_prototypes=200, delay_conv_stack_training=True)


def mnist():
    return dict(num_prototypes=20, conv_stack='mnist_conv')
