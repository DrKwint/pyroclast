from functools import lru_cache

import numpy as np
import sonnet as snt
import tensorflow as tf
from .weight_norm import WeightNorm
from .util import down_shift, right_shift, NIN, int_shape
import pyroclast.vqvae.nn as nn


def sigmoid_gating(x):
    """Apply the sigmoid gating in Figure 2 of [2]."""
    activation_tensor, gate_tensor = tf.split(x, 2, axis=-1)
    sigmoid_gate = tf.sigmoid(gate_tensor)
    return sigmoid_gate * activation_tensor


class GatedResBlock(tf.Module):

    def __init__(self,
                 num_filters,
                 kernel_shape=[3, 3],
                 padding='SAME',
                 conv=snt.Conv2D,
                 activation=tf.nn.elu,
                 dropout_p=0.1,
                 condition=False,
                 auxiliary=False):
        super().__init__()
        self.activation = activation
        self.conv1 = conv(num_filters,
                          kernel_shape=kernel_shape,
                          padding=padding)

        if auxiliary:
            self.aux_conv = NIN(num_filters)

        self.dropout = tf.keras.layers.Dropout(dropout_p)
        self.conv2 = WeightNorm(
            snt.Conv2D(num_filters * 2,
                       kernel_shape=kernel_shape,
                       padding=padding))

        if condition:
            self.condition_conv = conv(num_filters * 2, 1, with_bias=False)

        self.gate = sigmoid_gating

    def __call__(self, inputs, aux_input=None, condition=None):
        c1 = self.conv1(self.activation(inputs))
        if aux_input is not None:
            c1 += self.aux_conv(self.activation(aux_input))
        c1 = self.dropout(self.activation(c1))

        c2 = self.conv2(c1)
        if condition is not None:
            c2 += self.condition_conv(condition)

        c3 = self.gate(c2)
        return inputs + c3


@lru_cache(maxsize=64)
def causal_mask(size):
    shape = [size, size]
    mask = np.triu(np.ones(shape), k=1).astype(np.uint8).T
    start_mask = np.ones(size).astype(np.float32)
    start_mask[0] = 0

    return (
        tf.expand_dims(mask, 0),
        tf.expand_dims(start_mask, 1),
    )


class CausalAttention(tf.Module):

    def __init__(self, channel, n_head=8, dropout_p=0.1):
        super().__init__()

        self.query = WeightNorm(snt.Linear(channel))
        self.key = WeightNorm(snt.Linear(channel))
        self.value = WeightNorm(snt.Linear(channel))

        self.dim_head = channel // n_head
        self.n_head = n_head

        self.dropout = tf.keras.layers.Dropout(dropout)

    def __call__(self, query, key):
        batch, height, width, _ = key.shape

        def reshape(inputs):
            return tf.reshape(inputs, [batch, -1, self.n_head, self.dim_head])
            return inputs.view(batch, -1, self.n_head,
                               self.dim_head)  #.transpose(1, 2)

        raw = nn.nin(nn.gated_resnet(raw_content, conv=nn.nin), nr_filters // 2 + q_size)
                key, mixin = raw[:, :, :, :q_size], raw[:, :, :, q_size:]

        query_flat = tf.reshape(query,
                                [batch, query.shape[1], -1])  #.transpose(1, 2)
        key_flat = tf.reshape(key, [batch, key.shape[1], -1])  #.transpose(1, 2)
        query = reshape(self.query(query_flat))
        key = tf.transpose(reshape(self.key(key_flat)),
                           [0, 1, 3, 2])  # .transpose(2, 3)
        value = reshape(self.value(key_flat))
        attn = tf.matmul(query, key) / tf.math.sqrt(float(self.dim_head))
        mask, start_mask = causal_mask(height * width)
        mask = tf.cast(mask, query.dtype)
        start_mask = tf.cast(start_mask, query.dtype)
        print(mask.shape)
        print(attn.shape)
        exit()
        attn = attn.masked_fill(mask == 0, -1e4)
        attn = tf.nn.softmax(attn, 3) * start_mask
        attn = self.dropout(attn)

        out = attn @ value
        out = out.transpose(1, 2).reshape(batch, height, width,
                                          self.dim_head * self.n_head)
        out = out.permute(0, 3, 1, 2)

        return out


class PixelBlock(tf.Module):

    def __init__(
            self,
            channels,
            kernel_size,
            n_res_block,
            attention=True,
            dropout=0.1,
            condition=False,
    ):
        super().__init__()

        self.resblocks = []
        for _ in range(n_res_block):
            self.resblocks.append(
                GatedResBlock(
                    channels,
                    padding=snt.pad.causal,
                    dropout_p=dropout,
                    condition=condition,
                ))

        self.attention = attention

        if attention:
            q_size = 16
            self.key_layer = snt.Sequential([GatedResBlock(channels, conv=NIN)), NIN(channels // 2 + q_size)]
            self.key_resblock = GatedResBlock(channels * 2 + 2,
                                              1,
                                              dropout_p=dropout)
            self.query_resblock = GatedResBlock(channels + 2,
                                                1,
                                                dropout_p=dropout)

            self.causal_attention = CausalAttention(channels // 2,
                                                    dropout_p=dropout)

            self.out_resblock = GatedResBlock(
                channels,
                1,
                auxiliary=True,
                dropout=dropout,
            )

        else:
            self.out = WeightNorm(snt.Conv2D(channels, 1))

    def __call__(self, inputs, out, background, condition=None):
        for resblock in self.resblocks:
            out = resblock(out, condition=condition)

        if self.attention:
            key_cat = tf.concat([inputs, out, background], -1)
            raw = NIN(nn.gated_resnet(key_cat, conv=NIN), nr_filters // 2 + q_size)
            key = self.key_resblock(key_cat)
            query_cat = tf.concat([out, background], -1)
            query = self.query_resblock(query_cat)
            attn_out = self.causal_attention(query, key)
            out = self.out_resblock(out, attn_out)

        else:
            bg_cat = tf.concat([out, background], 1)
            out = self.out(bg_cat)

        return out


class PixelSNAIL(tf.Module):

    def __init__(
            self,
            shape,
            n_class,
            nr_filters,
            kernel_size,
            n_block,
            n_res_block,
            res_channel,
            attention=True,
            dropout=0.1,
            n_cond_res_block=0,
            cond_res_channel=0,
            cond_res_kernel=3,
            n_out_res_block=0,
    ):
        height, width, _ = shape

        self.n_class = n_class

        if kernel_size % 2 == 0:
            kernel = kernel_size + 1

        else:
            kernel = kernel_size

        self.horizontal = snt.Conv2D(nr_filters, [1, 3], padding=snt.pad.causal)
        self.vertical = snt.Conv2D(nr_filters, [2, 1], padding=snt.pad.causal)

        coord_x = (tf.cast(tf.range(height), tf.float32) - height / 2) / height
        coord_x = tf.tile(tf.reshape(coord_x, [1, height, 1, 1]),
                          [1, 1, width, 1])
        coord_y = (tf.cast(tf.range(width), tf.float32) - width / 2) / width
        coord_y = tf.tile(tf.reshape(coord_y, [1, 1, width, 1]),
                          [1, height, 1, 1])
        self.background = tf.concat([coord_x, coord_y], -1)

        self.blocks = []

        for i in range(n_block):
            self.blocks.append(
                PixelBlock(
                    nr_filters,
                    kernel_size,
                    n_res_block,
                    attention=attention,
                    dropout=dropout,
                    condition=cond_res_channel,
                ))

        if n_cond_res_block > 0:
            self.cond_resnet = CondResNet(n_class, cond_res_channel,
                                          cond_res_kernel, n_cond_res_block)

        out = []

        for i in range(n_out_res_block):
            out.append(GatedResBlock(channels, res_channel, 1))

        out.extend([tf.nn.elu, snt.Conv2D(n_class, 1)])

        self.out = snt.Sequential(out)

    def forward_loss(self, x):
        out, cache = self(x)
        print(out.shape)
        exit()

    def __call__(self, inputs, condition=None, cache=None):
        if cache is None:
            cache = {}
        batch, height, width = inputs.shape
        background = tf.tile(self.background, [batch, 1, 1, 1])

        x = inputs
        out = down_shift(self.horizontal(x)) + right_shift(self.vertical(x))

        if condition is not None:
            if 'condition' in cache:
                condition = cache['condition']
                condition = condition[:, :, :height, :]
            else:
                condition = tf.one_hot(condition, self.n_class)
                condition = self.cond_resnet(condition)
                condition = tf.image.resize(condition,
                                            2 * condition_size,
                                            method='nearest')
                cache['condition'] = condition.detach().clone()
                condition = condition[:, :, :height, :]

        for block in self.blocks:
            out = block(inputs, out, background, condition=condition)

        out = self.out(out)

        return out, cache
