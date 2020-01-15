import tensorflow as tf
from pyroclast.common.util import dummy_context_mgr
from tqdm import tqdm

from pyroclast.prototype.model import ProtoPNet
from pyroclast.common.models import get_network_builder


def learn(data_dict,
          seed,
          output_dir,
          debug,
          conv_stack='vgg19_conv',
          epochs=10,
          learning_rate=1e-3,
          cluster_coeff=0.8,
          l1_coeff=1e-4,
          separation_coeff=0.08,
          clip_norm=None,
          num_prototypes=20,
          prototype_dim=128,
          is_class_specific=False):
    writer = tf.summary.create_file_writer(output_dir)
    global_step = tf.compat.v1.train.get_or_create_global_step()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         beta_1=0.5,
                                         epsilon=0.01)

    # Using VGG19 here necessitates a 3 channel image input
    conv_stack = get_network_builder(conv_stack)()
    model = ProtoPNet(conv_stack,
                      num_prototypes,
                      prototype_dim,
                      data_dict['num_classes'],
                      class_specific=is_class_specific)

    # define minibatch fn
    def run_minibatch(epoch, batch, is_train=True):
        x = tf.cast(batch['image'], tf.float32) / 255.
        labels = tf.cast(batch['label'], tf.int32)

        with tf.GradientTape() if is_train else dummy_context_mgr() as tape:
            global_step.assign_add(1)
            y_hat, minimum_distances = model(x)
            classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=y_hat)
            loss_term_dict = model.conv_prototype_objective(minimum_distances,
                                                            label=labels)

            # build loss
            loss = classification_loss
            if 'cluster' in loss_term_dict:
                loss += cluster_coeff * loss_term_dict['cluster']
            if 'l1' in loss_term_dict:
                loss += l1_coeff * loss_term_dict['l1']
            if 'separation' in loss_term_dict:
                loss += separation_coeff * loss_term_dict['separation']
            loss = tf.reduce_mean(loss)

        # calculate gradients for current loss
        if is_train:
            gradients = tape.gradient(loss, model.trainable_variables)
            if clip_norm:
                clipped_gradients, pre_clip_global_norm = tf.clip_by_global_norm(
                    gradients, clip_norm)
            else:
                clipped_gradients = gradients
            optimizer.apply_gradients(
                zip(clipped_gradients, model.trainable_variables))

        # log to TensorBoard
        prefix = 'train_' if is_train else 'validate_'
        with writer.as_default():
            prediction = tf.math.argmax(y_hat, axis=1, output_type=tf.int32)
            classification_rate = tf.reduce_mean(
                tf.cast(tf.equal(prediction, labels), tf.float32))
            tf.summary.scalar(prefix + "classification_rate",
                              classification_rate,
                              step=global_step)
            tf.summary.scalar(prefix + "loss/mean classification",
                              tf.reduce_mean(classification_loss),
                              step=global_step)
            if 'cluster' in loss_term_dict:
                tf.summary.scalar(prefix + "loss/mean cluster",
                                  cluster_coeff *
                                  tf.reduce_mean(loss_term_dict['cluster']),
                                  step=global_step)
            if 'l1' in loss_term_dict:
                tf.summary.scalar(prefix + "loss/mean l1",
                                  l1_coeff *
                                  tf.reduce_mean(loss_term_dict['l1']),
                                  step=global_step)
            tf.summary.scalar(prefix + "loss/total loss",
                              loss,
                              step=global_step)

    # run training loop
    for epoch in range(epochs):
        # train
        train_batches = data_dict['train']
        if debug:
            print("Epoch", epoch)
            print("TRAIN")
            train_batches = tqdm(train_batches, total=data_dict['train_bpe'])
        for batch in train_batches:
            run_minibatch(epoch, batch, is_train=True)
