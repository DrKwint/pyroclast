import itertools
import os
import os.path as osp
import random
import subprocess

import luigi

from pyroclast.cpvae.cpvae import direct_eval, direct_learn
from pyroclast.luigi.util import get_base_path, limit_gpu_memory_usage

limit_gpu_memory_usage()


class CpVAETopTask(luigi.WrapperTask):

    def requires(self):
        return self.one_channel()

    def one_channel(self):
        arg_dict = {
            'dataset': ['mnist'],
            'encoder': ['mnist_encoder'],
            'decoder': ['mnist_decoder'],
            'latent_dim': [32, 64],
            'alpha': [1.],
            'beta': [1, 10.],
            'gamma': [100., 1000.],
            'output_dist': ['disc_logistic'],  #, 'l2'],  #, 'bernoulli'],
            'batch_size': [32, 64],
            'optimizer': ['rmsprop', 'adam'],
            'learning_rate': [3e-4],
            'oversample': [20, 50, 100],
            'max_tree_depth': [4, 6, 10],
            'tree_update_period': [2, 5],
            'clip_norm': [0.],
            'seed': [3745],
        }
        args = [
            dict(zip(arg_dict.keys(), arg_vals))
            for arg_vals in itertools.product(*arg_dict.values())
        ]
        random.shuffle(args)
        return [TrainTask(**a) for a in args]


class EvalTask(luigi.Task):
    dataset = luigi.Parameter()
    encoder = luigi.Parameter()
    decoder = luigi.Parameter()
    latent_dim = luigi.IntParameter()
    alpha = luigi.FloatParameter()
    beta = luigi.FloatParameter()
    gamma = luigi.FloatParameter()
    batch_size = luigi.IntParameter()
    output_dist = luigi.Parameter()  # disc_logistic or l2 or bernoulli
    batch_size = luigi.IntParameter()
    optimizer = luigi.Parameter()
    learning_rate = luigi.FloatParameter()
    oversample = luigi.IntParameter()
    max_tree_depth = luigi.IntParameter()
    max_tree_leaf_nodes = luigi.IntParameter()
    tree_update_period = luigi.IntParameter()
    clip_norm = luigi.FloatParameter()
    seed = luigi.IntParameter()

    priority = 1

    def requires(self):
        return TrainTask(
            dataset=self.dataset,
            encoder=self.encoder,
            decoder=self.decoder,
            latent_dim=self.latent_dim,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            output_dist=self.output_dist,  # disc_logistic or l2 or bernoulli
            batch_size=self.batch_size,
            optimizer=self.optimizer,  # adam or rmsprop
            learning_rate=self.learning_rate,
            oversample=self.oversample,
            max_tree_depth=self.max_tree_depth,
            max_tree_leaf_nodes=self.max_tree_leaf_nodes,
            tree_update_period=self.tree_update_period,
            clip_norm=self.clip_norm,
            seed=self.seed)

    def get_task_name_str(self):
        id_str = 'cpvae'
        id_str += '_ds{}'.format(self.dataset)
        id_str += '_enc{}'.format(self.encoder)
        id_str += '_dec{}'.format(self.decoder)
        id_str += '_ld{}'.format(self.latent_dim)
        id_str += '_alpha{}'.format(self.alpha)
        id_str += '_beta{}'.format(self.beta)
        id_str += '_gamma{}'.format(self.gamma)
        id_str += '_odist{}'.format(self.output_dist)
        id_str += '_bs{}'.format(self.batch_size)
        id_str += '_opt{}'.format(self.optimizer)
        id_str += '_lr{}'.format(self.learning_rate)
        id_str += '_os{}'.format(self.oversample)
        id_str += '_mtd{}'.format(self.max_tree_depth)
        id_str += '_mtn{}'.format(self.max_tree_leaf_nodes)
        id_str += '_tup{}'.format(self.tree_update_period)
        id_str += '_clip{}'.format(self.clip_norm)
        id_str += '_seed{}'.format(self.seed)
        return id_str

    def get_output_dir(self):
        return osp.join(get_base_path('cpvae'), self.get_task_name_str())

    def run(self):
        direct_eval(
            data_dict=self.dataset,
            encoder=self.encoder,
            decoder=self.decoder,
            seed=self.seed,
            latent_dim=self.latent_dim,
            epochs=1000,
            oversample=self.oversample,
            max_tree_depth=self.max_tree_depth,
            max_tree_leaf_nodes=self.max_tree_leaf_nodes,
            tree_update_period=self.tree_update_period,
            optimizer=self.optimizer,  # adam or rmsprop
            learning_rate=self.learning_rate,
            output_dist=self.output_dist,  # disc_logistic or l2 or bernoulli
            output_dir=osp.join(self.get_output_dir()),
            num_samples=5,
            clip_norm=self.clip_norm,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            patience=12,
            debug=True)

    def output(self):
        return luigi.LocalTarget(
            osp.join(self.get_output_dir(), 'final_loss.json'))


class TrainTask(luigi.Task):
    dataset = luigi.Parameter()
    encoder = luigi.Parameter()
    decoder = luigi.Parameter()
    latent_dim = luigi.IntParameter()
    alpha = luigi.FloatParameter()
    beta = luigi.FloatParameter()
    gamma = luigi.FloatParameter()
    output_dist = luigi.Parameter()  # disc_logistic or l2 or bernoulli
    batch_size = luigi.IntParameter()
    optimizer = luigi.Parameter()
    learning_rate = luigi.FloatParameter()
    oversample = luigi.IntParameter()
    max_tree_depth = luigi.IntParameter()
    tree_update_period = luigi.IntParameter()
    clip_norm = luigi.FloatParameter()
    seed = luigi.IntParameter()

    def get_output_dir(self):
        return osp.join(get_base_path('cpvae'), self.get_task_name_str())

    def get_task_name_str(self):
        id_str = 'cpvae'
        id_str += '_ds{}'.format(self.dataset)
        id_str += '_enc{}'.format(self.encoder)
        id_str += '_dec{}'.format(self.decoder)
        id_str += '_ld{}'.format(self.latent_dim)
        id_str += '_alpha{}'.format(self.alpha)
        id_str += '_beta{}'.format(self.beta)
        id_str += '_gamma{}'.format(self.gamma)
        id_str += '_odist{}'.format(self.output_dist)
        id_str += '_bs{}'.format(self.batch_size)
        id_str += '_opt{}'.format(self.optimizer)
        id_str += '_lr{}'.format(self.learning_rate)
        id_str += '_os{}'.format(self.oversample)
        id_str += '_mtd{}'.format(self.max_tree_depth)
        id_str += '_tup{}'.format(self.tree_update_period)
        id_str += '_clip{}'.format(self.clip_norm)
        id_str += '_seed{}'.format(self.seed)
        return id_str

    def run(self):
        """
        cmd_str = 'python -m pyroclast.run --module cpvae --task learn --patience 12 --debug'
        cmd_str += ' --dataset {}'.format(self.dataset)
        cmd_str += ' --encoder {}'.format(self.encoder)
        cmd_str += ' --decoder {}'.format(self.decoder)
        cmd_str += ' --seed {}'.format(self.seed)
        cmd_str += ' --latent_dim {}'.format(self.latent_dim)
        cmd_str += ' --batch_size {}'.format(self.batch_size)
        cmd_str += ' --epochs {}'.format(1000)
        cmd_str += ' --max_tree_depth {}'.format(self.max_tree_depth)
        cmd_str += ' --max_tree_leaf_nodes {}'.format(self.max_tree_leaf_nodes)
        cmd_str += ' --tree_update_period {}'.format(self.tree_update_period)
        cmd_str += ' --optimizer {}'.format(self.optimizer)
        cmd_str += ' --learning_rate {}'.format(self.learning_rate)
        cmd_str += ' --output_dist {}'.format(self.output_dist)
        cmd_str += ' --num_samples {}'.format(10)
        cmd_str += ' --clip_norm {}'.format(self.clip_norm)
        cmd_str += ' --alpha {}'.format(self.alpha)
        cmd_str += ' --beta {}'.format(self.beta)
        cmd_str += ' --gamma {}'.format(self.gamma)
        cmd_str += ' --patience {}'.format(12)
        cmd_str += ' --output_dir {}'.format(self.get_output_dir())
        subprocess.run(cmd_str.split(' '), check=True)
        """
        direct_learn(
            data_dict=self.dataset,
            encoder=self.encoder,
            decoder=self.decoder,
            seed=self.seed,
            latent_dim=self.latent_dim,
            batch_size=self.batch_size,
            epochs=1000,
            oversample=self.oversample,
            max_tree_depth=self.max_tree_depth,
            tree_update_period=self.tree_update_period,
            optimizer=self.optimizer,  # adam or rmsprop
            learning_rate=self.learning_rate,
            output_dist=self.output_dist,  # disc_logistic or l2 or bernoulli
            output_dir=self.get_output_dir(),
            num_samples=5,
            clip_norm=self.clip_norm,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            patience=12,
            debug=True)

    def output(self):
        return {
            "success":
                luigi.LocalTarget(osp.join(self.get_output_dir(), 'done'))
        }
