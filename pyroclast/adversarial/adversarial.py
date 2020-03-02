import importlib
from pyroclast.common.adversarial import fast_gradient_sign_method
from pyroclast.features.features import build_savable_objects


def visualize_perturbation(data_dict, seed, output_dir, debug, module_name,
                           model_name, norm, data_index, epsilon, **kwargs):
    module = importlib.import_module(module_name)
    model = load_model(module, model_name, data_dict, **kwargs)
    x = data_dict['train'][data_index]['image']
    y = data_dict['train'][data_index]['label']
    perturbed_x = fast_gradient_sign_method(x.features, x.logits, x, y, epsilon,
                                            norm)
    print("WOW")
