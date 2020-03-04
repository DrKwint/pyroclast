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
    y = tf.reshape(y, [1] + y.shape)

    num_classes = 10
    feature_idx = 0
    class_idx = 0

    def get_one_hot(x, num_classes):
        return tf.cast(tf.one_hot(x, num_classes, on_value=1, off_value=-1),
                       tf.float32)

    labels = get_one_hot(y, num_classes)

    forward_fn = lambda x: model.features(x)[feature_idx] * labels[:, class_idx]

    x_perturbation = fast_gradient_method(forward_fn, x, epsilon, norm)

    perturbed_x = x + x_perturbation

    plot_images([[x, x_perturbation, perturbed_x] for _ in range(3)],
                row_labels=['Images 1', 'Images 2', 'Images 3'],
                col_labels=['Original', 'Perturbation', 'Perturbed'])

    plt.show()
