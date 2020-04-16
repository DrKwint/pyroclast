import tensorflow as tf
from tqdm import tqdm
import os.path as osp
import tensorflow_datasets as tfds
import numpy as np


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


class PreprocessedDataset():

    def __init__(self,
                 ds,
                 module,
                 filepath,
                 data_key='image',
                 label_key='label'):
        self.module = module
        self.base_ds = ds

        if not osp.exists(filepath +
                          '_embeds.npy') or not osp.exists(filepath +
                                                           '_labels.npy'):
            self.save(filepath, data_key, label_key)
        self.ds = self.load(filepath, data_key, label_key)

    def __call__(self, batch_size):
        return self.ds.shuffle(1024).batch(batch_size)

    def save(self, base_path, data_key, label_key):

        def _inner(batch):
            embed = self.module(tf.cast(batch[data_key], tf.float32))
            label = batch[label_key]
            return (embed, label)

        ds = tfds.as_numpy(self.base_ds.map(_inner).unbatch())
        tuples = [x for x in tqdm(ds)]
        embeds, labels = zip(*tuples)
        np.save(base_path + '_embeds', np.array(embeds))
        np.save(base_path + '_labels', np.array(labels))

    def load(self, base_path, data_key, label_key):
        embeds = np.load(base_path + '_embeds.npy')
        labels = np.load(base_path + '_labels.npy')
        return tf.data.Dataset.from_tensors({
            data_key: embeds,
            label_key: labels
        }).unbatch()
