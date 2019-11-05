import os
import sys

import numpy as np
import sklearn
import tensorflow as tf
from flask import Flask, url_for
from PIL import Image

from pyroclast.cpvae.cpvae import CpVAE
from pyroclast.cpvae.tf_models import Decoder, Encoder
from pyroclast.cpvae.util import (build_model, setup_celeba_data,
                                  update_model_tree)

app = Flask(__name__, static_folder='static')
model = None


def learn(
        data_dict=None,
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
    # preprocess/load data
    data_dict = setup_celeba_data(batch_size, image_size)
    num_classes = 1

    # setup model vars
    tfvar_objs = build_model(optimizer_name=optimizer,
                             learning_rate=learning_rate,
                             num_classes=num_classes,
                             latent_dim=latent_dim,
                             image_size=image_size,
                             max_tree_depth=max_tree_depth,
                             max_tree_leaf_nodes=max_tree_leaf_nodes)

    # load trained model, if available
    checkpoint = tf.train.Checkpoint(**tfvar_objs)
    status = checkpoint.restore(tf.train.latest_checkpoint(str(load_dir)))
    print("load: ", status.assert_existing_objects_matched())

    # train a ddt
    update_model_tree(data_dict['train'],
                      tfvar_objs['model'],
                      epoch='visualize',
                      label_attr=label_attr,
                      output_dir=output_dir,
                      limit=10)

    return tfvar_objs


@app.route('/sample')
def sample():
    sample = np.squeeze(model.sample())
    im = Image.fromarray(((sample + 1) * 127.5).astype('uint8'), mode='RGB')
    im.save(os.path.join('./', "test.png"))

    sample_tensor = ((model.sample() + 1) * 127.5)
    sample_shape = sample_tensor.shape.as_list()
    sample_list = np.reshape(sample_tensor.numpy().astype('uint8'), [-1])
    return {'sample_shape': sample_shape, 'sample_values': sample_list.tolist()}


if __name__ == "__main__":
    model = learn()['model']
    app.run()
