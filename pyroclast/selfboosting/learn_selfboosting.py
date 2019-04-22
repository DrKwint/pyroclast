import sonnet as snt
import tensorflow as tf

from pyroclast import selfboosting
from pyroclast.selfboosting.build_graph import SequentialResNet
from pyroclast.common.tf_util import run_epoch_ops


def learn(train_data,
          seed,
          class_num=10,
          module_num=10,
          module_name='conv_block',
          representation_channels=32):

      model = SequentialResNet(class_num, representation_channels)


      final_classification, hypotheses, weak_classifiers = model(data['image'])
      with tf.Session() as session:
            module = model.boosting_modules[-1]
            model.get_hypothesis_loss(module.alpha, hypotheses[-1], data['label'])

    return None
