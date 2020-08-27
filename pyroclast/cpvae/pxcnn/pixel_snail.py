import tensorflow as tf


class PixelSnail(tf.Module):

    def __init__(self, n_class):
        self.horizontal = CausalConv2d(n_class,
                                       channel, [kernel // 2, kernel],
                                       padding='down')
        self.vertical = CausalConv2d(n_class,
                                     channel, [(kernel + 1) // 2, kernel // 2],
                                     padding='downright')

    def __call__(self):
        pass
