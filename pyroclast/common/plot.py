import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def plot_grads(images,
               models,
               model_names,
               image_shape,
               num_classes,
               debug=False,
               **kw):
    image_tensors = [[x] for x in tf.split(images, images.shape[0])]
    for j in range(len(models)):
        output = tf.split(models[j].certainty_sensitivity(images, num_classes),
                          images.shape[0])
        for i, o in enumerate(output):
            image_tensors[i].append(o)

    plot_images(image_tensors,
                col_labels=['original'] + model_names,
                cmap='seismic')
    plt.show()


def imshow_example(x, image_shape, **kwargs):
    image = tf.reshape(x, image_shape)
    imshow_kw = {'interpolation': 'none'}
    if len(image_shape) == 2:
        imshow_kw['cmap'] = 'gray'
    else:
        image = -image
    imshow_kw.update(kwargs)
    plt.xticks([])
    plt.yticks([])
    return plt.imshow(image, **imshow_kw)


def imshow_gradient(grad, image_shape, percentile=99, **kwargs):
    image = tf.reshape(grad, image_shape)
    if len(image_shape) == 3:
        # Convert RGB gradient to diverging BW gradient (ensuring the span isn't thrown off by outliers).
        # copied from https://github.com/PAIR-code/saliency/blob/master/saliency/visualization.py
        image = np.sum(image, axis=2)
        span = abs(np.percentile(image, percentile))
        vmin = -span
        vmax = span
        image = np.clip((image - vmin) / (vmax - vmin), -1, 1) * span
    imshow_kw = {'cmap': 'seismic', 'interpolation': 'none'}
    imshow_kw.update(kwargs)
    plt.xticks([])
    plt.yticks([])
    return plt.imshow(image, **imshow_kw)


def plot_images(image_tensors, row_labels=None, col_labels=None, cmap=None):
    """Plot images in a list or grid

    When this has run, the matplotlib.plt state is ready to be shown
    or saved to an image.

    Args:
       image_tensors (tf.Tensor | list(tf.Tensor) | list(list(tf.Tensor)): Tensors that hold image information. In order of (row, col).
       row_labels (None | list(str)): Labels for the rows
       col_labels (None | list(str)): Labels for the columns

    """

    if type(image_tensors) != list:
        image_tensors = [[image_tensors]]
    elif type(image_tensors[0]) != list:
        image_tensors = [image_tensors]

    nrows = len(image_tensors)
    ncols = max([len(x) for x in image_tensors])

    kwargs = {'cmap': cmap}

    if row_labels is not None:
        assert len(row_labels) == nrows
        for row_idx, label in enumerate(row_labels):
            index = row_idx * ncols + 1
            plt.subplot(nrows, ncols, index)
            side_text(label)

    if col_labels is not None:
        assert len(col_labels) == ncols
        for col_idx, label in enumerate(col_labels):
            index = col_idx + 1
            plt.subplot(nrows, ncols, index)
            top_text(label)

    for row_idx, row in enumerate(image_tensors):
        for col_idx, image_tensor in enumerate(row):
            index = row_idx * ncols + col_idx + 1
            axes = plt.subplot(nrows, ncols, index)
            squeezed_tensor = tf.squeeze(image_tensor)

            plt.xticks([])
            plt.yticks([])
            plt.imshow(squeezed_tensor, **kwargs)


def top_text(text, fontsize=8):
    left, width = 0.25, 0.5
    bottom, height = 1, 0.5
    right = left + width
    top = bottom + height
    ax = plt.gca()
    ax.text(0.5 * (left + right),
            bottom,
            text,
            ha='center',
            va='bottom',
            transform=ax.transAxes,
            fontsize=fontsize)


def side_text(text, fontsize=8):
    left, width = 0, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    ax = plt.gca()
    ax.text(left,
            0.5 * (bottom + top),
            text,
            ha='right',
            va='center',
            rotation='vertical',
            transform=ax.transAxes,
            fontsize=fontsize)
