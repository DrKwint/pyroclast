import tensorflow as tf
from pyroclast.common.util import dummy_context_mgr
from tqdm import tqdm

from pyroclast.prototype.model import ProtoPNet


def learn(data_dict,
          seed,
          output_dir,
          debug,
          epochs=10,
          learning_rate=1e-3,
          cluster_coeff=0.8,
          l1_coeff=1e-4,
          separation_coeff=0.08,
          clip_norm=None,
          num_prototypes=20,
          prototype_dim=128):
    global_step = tf.compat.v1.train.get_or_create_global_step()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         beta_1=0.5,
                                         epsilon=0.01)

    # Using VGG19 here necessitates a 3 channel image input
    conv_stack = tf.keras.applications.VGG19(include_top=False,
                                             pooling=None,
                                             input_shape=data_dict['shape'])
    model = ProtoPNet(conv_stack, num_prototypes, prototype_dim,
                      data_dict['num_classes'])

    # define minibatch fn
    def run_minibatch(epoch, batch, is_train=True):
        x = tf.cast(batch['image'], tf.float32) / 255.
        labels = tf.cast(batch['label'], tf.int32)

        with tf.GradientTape() if is_train else dummy_context_mgr() as tape:
            global_step.assign_add(1)
            y_hat, minimum_distances = model(x)
            classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=y_hat)
            loss_term_dict = model.conv_prototype_objective(minimum_distances)

            # build loss
            loss = classification_loss
            if 'cluster' in loss_term_dict:
                loss += cluster_coeff * loss_term_dict['cluster']
            if 'l1' in loss_term_dict:
                loss += l1_coeff * loss_term_dict['l1']
            if 'separation' in loss_term_dict:
                loss += separation_coeff * loss_term_dict['separation']

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
