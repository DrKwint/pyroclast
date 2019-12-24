import tensorflow as tf
from PIL import Image


class dummy_context_mgr():

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def center_crop(x, crop_size):
    # crop the images to [crop_size, crop_size, 3]
    h, w = x.shape.as_list()[:2]
    j = int(round((h - crop_size) / 2.))
    i = int(round((w - crop_size) / 2.))
    return x[j:j + crop_size, i:i + crop_size]


def img_preprocess(d, crop_size):
    x = center_crop(d['image'], crop_size)
    d['image'] = (tf.cast(x, tf.float32) / 127.5) - 1
    return d


def img_postprocess(x):
    return Image.fromarray(((x + 1) * 127.5).astype('uint8'), mode='RGB')
