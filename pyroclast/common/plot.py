import matplotlib.pyplot as plt
import numpy as np


def plot_grads(images, models, model_names, image_shape, **kw):
    fig = plt.figure(figsize=(len(images), len(images) + 1))
    for i in range(len(images)):
        plt.subplot(len(models) + 1, len(images), i + 1)
        imshow_example(images[i], image_shape)
        if i == 0: sidetext('Images')
    for j in range(len(models)):
        grads = models[j].certainty_sensitivity(images)
        for i in range(len(images)):
            plt.subplot(
                len(models) + 1, len(images), (j + 1) * len(images) + i + 1)
            imshow_gradient(grads[i], image_shape, **kw)
            if i == 0: sidetext(model_labels[model_names[j]])


model_labels = {
    'normal': 'Normal',
    'distilled': 'Distilled',
    'doubleback': r'Grad Reg ($y$)',
    'insensitive': r'Grad Reg ($\frac{1}{K}$)',
    'advtrain': 'Adv. Trained',
    'double-advtrain': 'Both Defenses',
    'double_advtrain': 'Both Defenses',
}


def sidetext(text, fontsize=8):
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


def imshow_example(x, image_shape, **kwargs):
    image = x.reshape(image_shape)
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
    image = grad.reshape(image_shape)
    if len(image_shape) == 3:
        # Convert RGB gradient to diverging BW gradient (ensuring the span isn't thrown off by outliers).
        # copied from https://github.com/PAIR-code/saliency/blob/master/saliency/visualization.py
        image = np.sum(image, axis=2)
        span = abs(np.percentile(image, percentile))
        vmin = -span
        vmax = span
        image = np.clip((image - vmin) / (vmax - vmin), -1, 1) * span
    imshow_kw = {'cmap': 'gray', 'interpolation': 'none'}
    imshow_kw.update(kwargs)
    plt.xticks([])
    plt.yticks([])
    return plt.imshow(image, **imshow_kw)
