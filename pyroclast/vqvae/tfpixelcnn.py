import pyroclast.vqvae.nn as nn
# from tensorflow.contrib.framework.python.ops import arg_scope
import tensorflow as tf


def model_spec(x,
               h=None,
               init=False,
               ema=None,
               dropout_p=0.5,
               nr_resnet=5,
               nr_filters=160,
               nr_logistic_mix=10,
               resnet_nonlinearity=tf.nn.elu):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    """
    with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense, nn.nin],
                   counters=counters,
                   init=init,
                   ema=ema,
                   dropout_p=dropout_p):

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity,
                       h=h):
"""
    # ////////// up pass through pixelCNN ////////
    xs = nn.int_shape(x)
    print(xs)
    background = tf.concat([
        ((tf.range(xs[1], dtype=tf.float32) - xs[1] / 2) /
         xs[1])[None, :, None, None] + 0. * x,
        ((tf.range(xs[2], dtype=tf.float32) - xs[2] / 2) /
         xs[2])[None, None, :, None] + 0. * x,
    ],
                           axis=3)
    print(background).shape
    # add channel of ones to distinguish image from padding later on
    x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)
    u_list = [
        nn.down_shift(
            nn.down_shifted_conv2d(x_pad,
                                   num_filters=nr_filters,
                                   filter_size=[2, 3]))
    ]  # stream for pixels above
    ul_list = [
        nn.down_shift(
            nn.down_shifted_conv2d(
                x_pad, num_filters=nr_filters, filter_size=[1, 3])) +
        nn.right_shift(
            nn.down_right_shifted_conv2d(
                x_pad, num_filters=nr_filters, filter_size=[2, 1]))
    ]  # stream for up and to the left

    for attn_rep in range(6):
        for rep in range(nr_resnet):
            u_list.append(
                nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
            ul_list.append(
                nn.gated_resnet(ul_list[-1],
                                u_list[-1],
                                conv=nn.down_right_shifted_conv2d))

        ul = ul_list[-1]

        raw_content = tf.concat([x, ul, background], axis=3)
        key, mixin = tf.split(nn.nin(nn.gated_resnet(raw_content, conv=nn.nin),
                                     nr_filters * 2),
                              2,
                              axis=3)
        query = nn.nin(
            nn.gated_resnet(tf.concat([ul, background], axis=3), conv=nn.nin),
            nr_filters)
        mixed = nn.causal_attention(key, mixin, query)

        ul_list.append(nn.gated_resnet(ul, mixed, conv=nn.nin))

    x_out = nn.nin(tf.nn.elu(ul_list[-1]), 10 * nr_logistic_mix)
    # x_out = nn.nin(tf.nn.elu(ul), 10 * nr_logistic_mix)

    # assert len(u_list) == 0
    # assert len(ul_list) == 0

    return x_out
