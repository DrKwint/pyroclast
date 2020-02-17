import tensorflow as tf


class PreprocessedDataset(object):

    def __init__(self, ds, module, filepath, key='image'):
        self.module = module
        self.base_ds = ds
        self.path = filepath

        # TODO: only save if it doesn't already exist
        # TODO: make filepath implicit by hashing the dataset and module
        self.save_tfrecord(key)
        self.ds = self.load_tfrecord()

    def __call__(self):
        return self.ds

    def save_tfrecord(self, key):
        # write to disk
        def inner(batch_dict):
            x = batch_dict[key]
            batch_dict[key] = self.module(x)

        writer = tf.data.experimental.TFRecordWriter(self.path)
        writer.write(self.base_ds.map(inner))

    def load_tfrecord(self):
        raw_dataset = tf.data.TFRecordDataset([self.path])
        self.ds = raw_dataset
