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

    print('x', x.shape)

    pred = model(x)
    print('pred', pred)

    pred_idx = tf.argmax(pred, axis=1)
    print('pred_idx', pred_idx)

    asdf = tf.gather(pred, pred_idx, axis=1)
    print(asdf)
    num_classes = 10
    feature_idx = 0
    class_idx = 0

    def get_one_hot(x, num_classes):
        return tf.cast(tf.one_hot(x, num_classes, on_value=1, off_value=-1),
                       tf.float32)

    labels = get_one_hot(y, num_classes)
    print('y', y)

    # Only works for GenericClassifier
    for v in model.classifier.trainable_variables:
        if 'kernel' in v.name:
            print('shape', v.shape)
            class_slice = v[:, y]
            print('slice', class_slice)
            print(class_slice.shape)
            m = tf.argmax(class_slice)
            print(m)
            break

    class_idx = y.numpy()
    print('class', class_idx)
    feature_idx = m.numpy()
    print('feature', feature_idx)

    print('f_slice', model.features(x)[:, feature_idx])
    print('l_slice', labels[class_idx])

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
    print('labels', len(col_labels))

    print('perturbed', len(perturbed))
    print('perturbations', len(perturbations))
    plot_images([perturbed, perturbations],
                row_labels=row_labels,
                col_labels=col_labels)

    plt.show()
