import functools
import itertools
import json
import os.path as osp
import pathlib
import random
import shutil
import subprocess

import luigi

from pyroclast.luigi.util import get_base_path, limit_gpu_memory_usage

# limit_gpu_memory_usage()


class FeaturesTopTask(luigi.WrapperTask):

    def requires(self):
        if not osp.exists(get_base_path('features')):
            raise Exception('base path {} not accessible'.format(
                get_base_path('features')))
        return self.train_tasks()

    def one_channel_train_tasks(self):
        arg_dict = {
            'dataset': ['mnist'],
            'batch_size': [128],
            'seed': [8549],
            'learning_rate': [1e-4],
            'conv_stack_name': ['attack_net'],
            'lambd': [0.]
        }
        args = [
            dict(zip(arg_dict.keys(), arg_vals))
            for arg_vals in itertools.product(*arg_dict.values())
        ]
        return [TrainTask(**a) for a in args]

    def three_channel_train_tasks(self):
        arg_dict = {
            'dataset': ['cifar10'],
            'batch_size': [128],
            'seed': [8549],
            'learning_rate': [1e-4],
            'conv_stack_name': ['attack_net', 'vgg19'],
            'lambd': [0.]
        }
        args = [
            dict(zip(arg_dict.keys(), arg_vals))
            for arg_vals in itertools.product(*arg_dict.values())
        ]
        return [TrainTask(**a) for a in args]

    def train_tasks(self):
        return self.one_channel_train_tasks() + self.three_channel_train_tasks()


class TrainTask(luigi.Task):
    dataset = luigi.Parameter()
    batch_size = luigi.IntParameter()
    seed = luigi.IntParameter()

    learning_rate = luigi.FloatParameter()
    conv_stack_name = luigi.Parameter()
    lambd = luigi.FloatParameter()

    def get_task_name_str(self):
        id_str = 'learn'
        id_str += '_{}'.format(self.dataset)
        id_str += '_{}'.format(self.conv_stack_name)
        id_str += '_{}'.format(self.batch_size)
        id_str += '_{}'.format(self.learning_rate)
        id_str += '_{}'.format(self.lambd)
        id_str += '_{}'.format(self.seed)
        return id_str

    def get_output_dir(self):
        return osp.join(get_base_path('features'), self.get_task_name_str())

    def run(self):
        local_output_dir = osp.join('./tmp', self.get_task_name_str())
        remote_output_dir = self.get_output_dir()

        cmd_str = 'python -m pyroclast.run --module features --task learn --patience 12 --max_epochs 200'
        cmd_str += ' --dataset {}'.format(self.dataset)
        cmd_str += ' --conv_stack_name {}'.format(self.conv_stack_name)
        cmd_str += ' --batch_size {}'.format(self.batch_size)
        cmd_str += ' --learning_rate {}'.format(self.learning_rate)
        cmd_str += ' --seed {}'.format(self.seed)
        cmd_str += ' --output_dir {}'.format(local_output_dir)
        subprocess.run(cmd_str.split(' '), check=True)
        shutil.copytree(local_output_dir, remote_output_dir)

    def output(self):
        print(self.get_output_dir())
        return {
            "success":
                luigi.LocalTarget(osp.join(self.get_output_dir(), 'done')),
            "output_dir":
                luigi.LocalTarget(self.get_output_dir())
        }
