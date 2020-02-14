def caltech_birds2011():
    return dict(num_prototypes=200, delay_conv_stack_training=True)


def mnist():
    return dict(max_epochs_phase_1=20,
                patience_phase_1=2,
                max_epochs_phase_3=20,
                patience_phase_3=3,
                num_prototypes=100,
                conv_stack='mnist_conv',
                is_class_specific=True,
                prototype_dim=64)
