import os

import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


class dummy_context_mgr():

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def center_crop(x, crop_size):
    # crop the images to [crop_size, crop_size, 3]
    h, w = x.shape.as_list()[:2]
    j = int(round((h - crop_size) / 2.))
    i = int(round((w - crop_size) / 2.))
    return x[j:j + crop_size, i:i + crop_size]


def img_preprocess(d, crop_size):
    x = center_crop(d['image'], crop_size)
    d['image'] = (tf.cast(x, tf.float32) / 127.5) - 1
    return d


def img_postprocess(x):
    if len(x.shape) == 3 and x.shape[-1] == 3:
        return Image.fromarray(((x + 1) * 127.5).astype('uint8'), mode='RGB')
    else:
        return Image.fromarray(((x + 1) * 127.5).astype('uint8'))


def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def heatmap(matrix, path, title):
    X = np.hstack([
        matrix.numpy(),
        np.reshape(np.arange(matrix.shape[0]), [matrix.shape[0], 1])
    ])
    for i in range(matrix.shape[-1]):
        X = X[X[:, i].argsort()]
    matrix = X[:, :-1]
    idxs = X[:, -1]

    fig, ax = plt.subplots(figsize=(matrix.shape[1] // 2, matrix.shape[0] // 2))
    divnorm = colors.DivergingNorm(vcenter=0)
    plot = ax.pcolormesh(matrix, cmap='seismic', norm=divnorm)
    ax.set_yticks(np.arange(len(idxs)))
    ax.set_yticklabels(idxs)
    fig.colorbar(plot)
    ax.set_title(title)
    plt.savefig(path)
    plt.close(fig)
