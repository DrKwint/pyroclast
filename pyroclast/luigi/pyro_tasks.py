import luigi
import functools
import itertools
import json
import random

from pyroclast.luigi.train_task import TrainTask


class TopTask(luigi.Task):

    def requires(self):
        return self.train_tasks()

    def train_tasks(self):
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

    def complete(self):
        return False
