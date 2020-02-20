import os

import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


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
    fig, ax = plt.subplots(figsize=(matrix.shape[1], matrix.shape[0]))
    plot = ax.pcolormesh(matrix, cmap='hot')
    fig.colorbar(plot)
    # We want to show all ticks...
    # ax.set_xticks(np.arange(len(vipd[0])))
    # ax.set_yticks(np.arange(len(variable_names)))
    # ... and label them with the respective list entries
    # ax.set_xticklabels(range(len(vipd[0])))
    # ax.set_yticklabels(variable_names)
    # Loop over data dimensions and create text annotations.
    #for i in range(len(variable_names)):
    #    for j in range(len(vipd[0])):
    #        ax.text(j, i, f’{vipd[i,j]:,}’, ha=‘center’, va=‘center’)
    ax.set_title(title)
    plt.savefig(path)
    plt.close(fig)
