import luigi
import functools
import itertools
import json
import random
import os.path as osp


class TopTask(luigi.WrapperTask):

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
        return osp.join(self.base_path, self.get_task_name_str())

    def run(self):
        cmd_str = 'python -m pyroclast.run --module features --task learn --patience 12 --max_epochs 200'
        cmd_str += ' --dataset {}'.format(self.dataset)
        cmd_str += ' --conv_stack_name {}'.format(self.conv_stack_name)
        cmd_str += ' --batch_size {}'.format(self.batch_size)
        cmd_str += ' --learning_rate {}'.format(self.learning_rate)
        cmd_str += ' --seed {}'.format(self.seed)
        cmd_str += ' --lambd {}'.format(self.lambd)
        subprocess.run(cmd_str.split(' '), check=True)

    def output(self):
        return {
            "success":
                luigi.LocalTarget(osp.join(self.get_output_dir(), 'done'))
        }
