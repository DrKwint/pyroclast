import tensorflow as tf
from tqdm import tqdm


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


class PreprocessedDataset(object):

    def __init__(self,
                 ds,
                 module,
                 filepath,
                 data_key='image',
                 label_key='label'):
        self.module = module
        self.base_ds = ds
        self.path = filepath

        # TODO: only save if it doesn't already exist
        # TODO: make filepath implicit by hashing the dataset and module
        self.save_tfrecord(data_key, label_key)
        self.ds = self.load_tfrecord()

    def __call__(self):
        return self.ds

    def save_tfrecord(self, data_key, label_key):
        # write to disk
        def serialize_example(data, label):
            embed = self.module(tf.cast(data, tf.float32))

            feature = {
                'data': _float_feature(embed),
                'label': _int64_feature(label)
            }

            example_proto = tf.train.Example(features=tf.train.Features(
                feature=feature))
            return example_proto.SerializeToString()

        def tf_serialize_example(d, l):
            tf_string = tf.py_function(
                serialize_example,
                (d, l),  # pass these args to the above function.
                tf.string)  # the return type is `tf.string`.
            return tf.reshape(tf_string, ())  # The result is a scalar

        #serialized_features_dataset = self.base_ds.map(
        #    lambda x: tf_serialize_example(x[data_key], x[label_key]))
        writer = tf.data.experimental.TFRecordWriter(self.path)
        with tf.io.TFRecordWriter(self.path) as writer:
            for x in tqdm(self.base_ds.unbatch()):
                example = serialize_example(tf.expand_dims(x[data_key], 0),
                                            tf.expand_dims(x[label_key], 0))
                writer.write(example)

    def load_tfrecord(self):
        raw_dataset = tf.data.TFRecordDataset([self.path])
        self.ds = raw_dataset
