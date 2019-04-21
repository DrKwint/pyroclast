import sonnet as snt
import tensorflow as tf

from pyroclast import selfboosting
from pyroclast.selfboosting.build_graph import SequentialResNet


def learn(train_data,
          seed,
          class_num=10,
          module_num=10,
          module_name='conv_block',
          representation_channels=32):

    model = SequentialResNet(class_num, representation_channels)

    return None
