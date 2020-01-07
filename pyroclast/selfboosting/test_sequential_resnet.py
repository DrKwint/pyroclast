# import numpy as np
# import pytest
# import tensorflow as tf

# from pyroclast.selfboosting.residual_boosting_module import ResidualBoostingModule
# from pyroclast.selfboosting.sequential_resnet import SequentialResNet

# @pytest.mark.skip(reason="Old TF1 code")
# class TestSequentialResNet(object):
#     # random values
#     batch_size = np.random.randint(low=1, high=66)
#     data_shape = [np.random.randint(low=1, high=128) for _ in range(3)]
#     num_classes = np.random.randint(low=2, high=24)
#     num_channels = np.random.randint(low=1, high=75)

#     def setup_tf_env(self):
#         tf.reset_default_graph()
#         session = tf.Session()
#         data_ph = tf.placeholder(shape=[self.batch_size] + self.data_shape,
#                                  dtype=tf.float32,
#                                  name='data_placeholder')
#         label_ph = tf.placeholder(shape=[self.batch_size],
#                                   dtype=tf.int64,
#                                   name='label_placeholder')
#         model = SequentialResNet(self.num_classes, self.num_channels)
#         return session, model, data_ph, label_ph

#     def setup_residual_boosting_module(self):
#         import random
#         repr_module_name = random.choice(["conv_block"])
#         hypothesis_module_name = random.choice(["linear_classifier"])

#         return ResidualBoostingModule(repr_module_name, hypothesis_module_name,
#                                       self.num_channels, self.num_classes)

#     @pytest.mark.skip(reason="Old TF1 code")
#     def test_add_before_build_fails(self):
#         _, model, _, _ = self.setup_tf_env()
#         module = self.setup_residual_boosting_module()
#         with pytest.raises(snt.NotConnectedError):
#             model.add_module(module)

#     @pytest.mark.skip(reason="Old TF1 code")
#     def test_only_trains_one_module(self):
#         """Takes about 1 minute to run"""
#         session, model, data_ph, label_ph = self.setup_tf_env()
#         optimizer = tf.train.AdamOptimizer(1e-5)
#         num_blocks = np.random.randint(low=2, high=5)

#         model(data_ph, is_train=True)
#         model_variables = model.get_all_variables()
#         module_variables = []
#         module_train_ops = []
#         for _ in range(num_blocks):
#             module = self.setup_residual_boosting_module()
#             alpha, hypothesis, _ = model.add_module(module)
#             module_variables.append(module.get_all_variables())
#             module_loss = model.get_hypothesis_loss(alpha, hypothesis, label_ph)
#             module_train_ops.append(module.get_train_op(optimizer, module_loss))

#         session.run(tf.initializers.global_variables())
#         stable_model_values = session.run(model_variables)
#         stable_module_values = session.run(module_variables)
#         approx_pair = lambda tup: tup[0] == pytest.approx(tup[1])
#         approx_list = lambda stable, variables: all(
#             map(approx_pair, zip(stable, session.run(variables))))
#         for i, train_op in enumerate(module_train_ops):
#             session.run(
#                 train_op, {
#                     data_ph:
#                         np.random.random([self.batch_size] + self.data_shape),
#                     label_ph:
#                         np.random.randint(low=0,
#                                           high=self.num_classes - 1,
#                                           size=self.batch_size)
#                 })
#             if not approx_list(stable_model_values, model_variables):
#                 raise Exception(
#                     "Model variables got changed while training module", i)
#             for j in range(num_blocks):
#                 if j == i: continue
#                 if not approx_list(stable_module_values[j],
#                                    module_variables[j]):
#                     raise Exception(
#                         "Module {} variables got changed while training module {}"
#                         .format(j, i))
#             # update stable values
#             stable_module_values[i] = session.run(module_variables[i])