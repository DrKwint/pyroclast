from pyroclast.cpvae.learn import setup
from pyroclast.cpvae.util import calculate_celeba_latent_values, get_node_members

import tensorflow as tf


def learn(
        data_dict,
        seed=None,
        latent_dim=128,
        epochs=1000,
        batch_size=64,
        image_size=128,
        max_tree_depth=5,
        max_tree_leaf_nodes=16,
        tree_update_period=10,
        label_attr='No_Beard',
        optimizer='adam',  # adam or rmsprop
        learning_rate=1e-3,
        classification_coeff=1.,
        distortion_fn='disc_logistic',  # disc_logistic or l2
        output_dir='./',
        load_dir=None):
    model, optimizer, global_step, writer, ckpt_manager = setup(**locals())
    for i in range(max_tree_leaf_nodes):
        node_ds = get_node_members(data_dict['train'], model, i)
        locs, scales, samples, attrs = calculate_celeba_latent_values(
            node_ds, model, label_attr)

        # using gaussian convolution
        # from http://www.tina-vision.net/docs/memos/2003-003.pdf
        gaussian_conv_variance = 1. / tf.reduce_sum(1. / tf.math.square(scales),
                                                    axis=1)
        gaussian_conv_loc = tf.reduce_sum(locs / scales,
                                          axis=1) * gaussian_conv_variance
        gaussian_conv_scale = tf.math.sqrt(gaussian_conv_variance)

        # using loc as a point estimate
        # from http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
        gaussian_mle_loc = tf.reduce_mean(locs, axis=1)
        gaussian_mle_scale = tf.reduce_mean(tf.math.square(locs -
                                                           gaussian_mle_loc),
                                            axis=1)
        print("Node", i)
        print("Convolution loc:", gaussian_conv_loc)
        print("Convolution scale:", gaussian_conv_loc)
        print("Gaussian MLE loc:", gaussian_mle_loc)
        print("Gaussian MLE loc:", gaussian_mle_scale)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('load_dir')
