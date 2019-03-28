import sonnet as snt
import tensorflow as tf


def conv_block(channels):
    return snt.Sequential([
        snt.Conv2D(channels, 3), tf.nn.relu,
        snt.Conv2D(channels, 3), tf.nn.relu
    ])


modules = {'conv_block': conv_block}


class ResNet(snt.Module):
    def __init__(self, initial_trans, modules):
        self._initial_trans = initial_trans
        self._res_modules = [snt.Residual(m) for m in modules]

    def _all_reprs(self, inputs):
        x = self._initial_trans(inputs)
        reprs = [x]
        for module in self._res_modules:
            x = module(x)
            reprs.append(x)
        return reprs

    def _build(self, inputs):
        return self._all_reprs(inputs)[-1]
