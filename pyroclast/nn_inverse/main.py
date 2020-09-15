import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)



ds = tfds.load('imagenette', split='train', shuffle_files=True)
for x in ds:
    img = x['image']
    label = x['label']
    img = np.expand_dims(img, 0)
    x = tf.keras.applications.vgg19.preprocess_input(
    img
)
    block4_pool_features = model.predict(x)
    print(block4_pool_features.shape)
