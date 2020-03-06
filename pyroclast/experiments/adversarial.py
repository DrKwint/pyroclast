import importlib
import tensorflow as tf
import matplotlib.pyplot as plt

from pyroclast.common.tf_util import load_model
from pyroclast.common.plot import plot_images
from pyroclast.common.adversarial import fast_gradient_method
from pyroclast.features.features import build_savable_objects


def visualize_perturbation(data_dict, seed, output_dir, debug, module_name,
                           model_name, norm, data_index, epsilon, **kwargs):
    module = importlib.import_module(module_name)
    model = load_model(module,
                       model_name,
                       data_dict,
                       output_dir=output_dir,
                       **kwargs)
    for batch_data in data_dict['train']:
        x = batch_data['image'][0]
        y = batch_data['label'][0]
        break
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [1] + x.shape)

    pred = model(x)
    pred_idx = tf.argmax(pred, axis=1)

    y = tf.reshape(y, [1] + y.shape)

    num_classes = 10
    feature_idx = 0
    class_idx = 0

    def get_one_hot(x, num_classes):
        return tf.cast(tf.one_hot(x, num_classes, on_value=1, off_value=-1),
                       tf.float32)

    labels = get_one_hot(y, num_classes)

    forward_fn = lambda x: -tf.gather(model(x), pred_idx, axis=1)

    epsilons = [0.01, 0.02, 0.03]
    # epsilons = [0.01, 0.1, 1]
    # epsilons = [0.01 * x for x in range(1, 11)]

    perturbations = [tf.zeros(x.shape)] + [
        fast_gradient_method(forward_fn, x, eps, norm) for eps in epsilons
    ]
    perturbed = [x + pert for pert in perturbations]
    perturbations = [p * 16 for p in perturbations]

    perturbed_tensor = tf.concat(perturbed, 0)
    preds = model(perturbed_tensor)
    print('Pred idx', pred_idx)
    print('Initial prediction', pred.numpy())
    print('Perturb prediction', preds.numpy())
    print('Perturb prediction idx', tf.argmax(preds, axis=1))

    row_labels = ['Image', 'Perturbation']
    col_labels = ['Original'] + [str(eps) for eps in epsilons]

    plot_images([perturbed, perturbations],
                row_labels=row_labels,
                col_labels=col_labels)

    plt.show()


def visualize_smoothgrad(data_dict, seed, output_dir, debug, module_name,
                         model_name, data_index, **kwargs):
    module = importlib.import_module(module_name)
    model = load_model(module, model_name, data_dict, output_dir=output_dir)
    for batch_data in data_dict['train']:
        x = batch_data['image'][0]
        y = batch_data['label'][0]
        break
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [1] + x.shape)

    smooth_grad = model.smooth_grad(x)
    gray_scale_image = tf.image.rgb_to_grayscale(smooth_grad)
    plot_images([x, gray_scale_image], cmap='gray')
    plt.show()
