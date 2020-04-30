import itertools
import json
import os
from pathlib import Path

import luigi
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from pyroclast.common.tf_util import setup_tfds
from pyroclast.features.features import build_savable_objects
from pyroclast.luigi.features_tasks import TrainTask
from pyroclast.luigi.util import get_base_path

if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
    import json


def f_float(f):
    return str(f).replace('.', '-')


def jnd_norm():
    pass


def get_norm(norm_str):
    if norm_str == 'jnd':
        return jnd_norm()
    if norm_str == 'l_1':
        return 1
    if norm_str == 'l_2':
        return 2
    if norm_str == 'l_inf':
        return np.inf
    raise Exception(f'Unknown norm string: {norm_str}')


class robustness_curves(luigi.Config):
    storage_dir = luigi.Parameter(default='./robustness')
    norm = luigi.ChoiceParameter(choices=['jnd', 'l_2', 'l_1', 'l_inf'],
                                 default='l_2')
    epsilons = luigi.ListParameter(default=[0.0, 1.0])
    model_name = luigi.Parameter(default='generic_classifier')
    model_params = luigi.DictParameter(
        default={
            'dataset': 'mnist',
            'batch_size': 128,
            'seed': 8549,
            'learning_rate': 1e-4,
            'conv_stack_name': 'attack_net',
            'lambd': 0.
        })


data_dict_singleton = None
model_objects_singleton = None
num_features_singleton = None


def get_data_dict():
    global data_dict_singleton
    if data_dict_singleton is None:
        config = robustness_curves()
        data_dict_singleton = setup_tfds(
            config.model_params['dataset'],
            config.model_params['batch_size'],
            None,
            -1,
            None,
            shuffle_seed=config.model_params['seed'])
    return data_dict_singleton


def get_model_objects():
    global model_objects_singleton
    if model_objects_singleton is None:
        config = robustness_curves()
        data_dict = get_data_dict()
        model_objects_singleton = build_savable_objects(
            config.model_params['conv_stack_name'],
            data_dict, config.model_params['learning_rate'],
            get_base_path('features'), config.model_name)
    return model_objects_singleton


def get_num_classes():
    return get_data_dict()['num_classes']


def get_num_features():
    global num_features_singleton
    if num_features_singleton is None:
        config = robustness_curves()
        objects = get_model_objects()
        model = objects['model']
        checkpoint = objects['checkpoint']
        checkpoint.restore(tf.train.latest_checkpoint(
            get_base_path('features')))
        data_dict = get_data_dict()
        for d in data_dict['train']:
            num_features_singleton = model.features(
                tf.cast(d['image'], tf.float32)).shape[-1]
            break
    return num_features_singleton


def specs():
    classes_iter = range(get_num_classes())
    features_iter = range(get_num_features())
    epsilons = robustness_curves().epsilons
    iterators = [classes_iter, features_iter, epsilons]
    return itertools.product(*iterators)


class TopTask(luigi.Task):

    def output(self):
        config = robustness_curves()
        return [
            luigi.LocalTarget(
                Path(config.storage_dir) / 'images' /
                (f'curves-{class_idx}.png'))
            for class_idx in range(get_num_classes())
        ]

    def requires(self):
        return [
            RobustnessTask(class_idx=class_idx,
                           feature_idx=feature_idx,
                           eps=eps)
            for (class_idx, feature_idx, eps) in specs()
        ]

    def run(self):
        lines = [[{
            'x': [],
            'y': []
        } for _ in range(get_num_features())] for _ in range(get_num_classes())]

        for (class_idx, feature_idx,
             eps), input_target in zip(specs(), self.input()):
            with input_target.open('r') as in_file:
                robustness = json.load(in_file)
            lines[class_idx][feature_idx]['x'].append(eps)
            lines[class_idx][feature_idx]['y'].append(robustness)

        for class_idx, out_target in zip(range(get_num_classes()),
                                         self.output()):
            with out_target.open('w') as out_file:
                fig, ax1 = plt.subplots()
                ax1.set_xlabel('Epsilon')
                ax1.set_xscale('log')
                ax1.set_ylabel('Usefulness')
                min_x = min([
                    min([x
                         for x in feature_line['x']
                         if x > 0])
                    for feature_line in lines[class_idx]
                ])
                for feature_line in lines[class_idx]:
                    if feature_line['y'][0] > 0:
                        if feature_line['x'][0] == 0:
                            feature_line['x'][0] = min_x / 10
                        plt.plot(feature_line['x'], feature_line['y'])
                plt.savefig(out_file.tmp_path, format='png')
                plt.close(fig)


class RobustnessTask(luigi.Task):
    class_idx = luigi.IntParameter()
    feature_idx = luigi.IntParameter()
    eps = luigi.FloatParameter()

    def requires(self):
        config = robustness_curves()
        return TrainTask(**config.model_params)

    def output(self):
        config = robustness_curves()
        storage_dir = Path(config.storage_dir)
        file_name = f'robustness_{self.class_idx}_{self.feature_idx}_{f_float(self.eps)}.json'
        return luigi.LocalTarget(storage_dir / 'values' / file_name)

    def run(self):
        config = robustness_curves()

        # load dataset
        data_dict = setup_tfds(config.model_params['dataset'],
                               config.model_params['batch_size'],
                               None,
                               -1,
                               None,
                               shuffle_seed=config.model_params['seed'])
        num_classes = data_dict['num_classes']

        # load model
        objects = build_savable_objects(config.model_params['conv_stack_name'],
                                        data_dict,
                                        config.model_params['learning_rate'],
                                        self.input()['output_dir'].path,
                                        config.model_name)

        model = objects['model']
        checkpoint = objects['checkpoint']
        ckpt_manager = objects['ckpt_manager']
        checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
        data_map = data_dict['train'].map(
            lambda x: (tf.cast(x['image'], tf.float32), x['label']))
        if self.eps == 0:
            robustness = model.usefulness(
                data_map, num_classes)[self.feature_idx][self.class_idx]
        else:
            robustness = model.robustness(data_map, self.feature_idx,
                                          self.class_idx, num_classes, self.eps,
                                          get_norm(config.norm))
        robustness = robustness.numpy().tolist()

        with self.output().open('w') as out_file:
            json.dump(robustness, out_file)


if __name__ == '__main__':
    luigi.build([TopTask()])
