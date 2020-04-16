import tensorflow as tf


def build_savable_objects(conv_stack_name, data_dict, learning_rate, model_dir,
                          model_name):
    global_step = tf.compat.v1.train.get_or_create_global_step()
    if conv_stack_name == 'vgg19':
        conv_stack = get_network_builder(conv_stack_name)(shape=[32, 32, 3])
    else:
        conv_stack = get_network_builder(conv_stack_name)()
    classifier = tf.keras.Sequential(
        [tf.keras.layers.Dense(data_dict['num_classes'])])

    clean = lambda varStr: re.sub('\W|^(?=\d)', '_', varStr)
    model = GenericClassifier(conv_stack, classifier, clean(model_name))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         beta_1=0.5,
                                         epsilon=10e-4)
    save_dict = {
        model_name + '_optimizer': optimizer,
        model_name + '_model': model,
        model_name + '_global_step': global_step
    }
    checkpoint = tf.train.Checkpoint(**save_dict)

    ckpt_manager = tf.train.CheckpointManager(checkpoint,
                                              directory=os.path.join(
                                                  model_dir, model_name),
                                              max_to_keep=3)
    return {
        'model': model,
        'optimizer': optimizer,
        'global_step': global_step,
        'checkpoint': checkpoint,
        'ckpt_manager': ckpt_manager
    }


def calculate_loss(dataset, model, num_classes, lambd, alpha):
    loss_numerator = 0
    loss_denominator = 0
    dummy_var = tf.Variable(initial_value=0)
    for batch in dataset:
        l, _, d = run_minibatch(model,
                                optimizer=None,
                                global_step=dummy_var,
                                epoch=0,
                                batch=batch,
                                num_classes=num_classes,
                                lambd=lambd,
                                alpha=alpha,
                                writer=None,
                                is_train=False)
        loss_numerator += l
        loss_denominator += d
    return loss_numerator / float(loss_denominator)


# define minibatch fn
@tf.function
def run_minibatch(model,
                  optimizer,
                  global_step,
                  epoch,
                  batch,
                  num_classes,
                  lambd,
                  alpha,
                  writer=None,
                  is_train=True):
    """
    Args:
        model (tf.Module):
        optimizer (tf.Optimizer):
        global_step (Tensor):
        epoch (int): Epoch of training for logging
        batch (dict): dict from dataset
        writer (tf.summary.SummaryWriter):
        is_train (bool): Optional, run backwards pass if True
    """
    x = tf.cast(batch['image'], tf.float32) / 255.
    labels = tf.cast(batch['label'], tf.int32)
    with tf.GradientTape() as tape:
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(x)
            y_hat = model(x)
            prediction = tf.math.argmax(y_hat, axis=1, output_type=tf.int32)

            # classification loss
            classification_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(labels, num_classes), logits=y_hat)

        if lambd != 0.:
            # input gradient regularization
            grad = inner_tape.gradient(y_hat, x)
            input_grad_reg_loss = tf.math.square(tf.norm(grad, 2))

            if alpha != 0.:
                grad_masked_y_hat = model(x * grad)
                grad_masked_classification_loss = tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.one_hot(labels, num_classes),
                    logits=grad_masked_y_hat)
            else:
                grad_masked_classification_loss = 0.
        else:
            input_grad_reg_loss = 0.
            grad_masked_classification_loss = 0.

        # total loss
        total_loss = classification_loss + (lambd * input_grad_reg_loss) + (
            alpha * grad_masked_classification_loss)
        mean_total_loss = tf.reduce_mean(total_loss)
        global_step.assign_add(1)

    if is_train:
        train_vars = model.trainable_variables
        gradients = tape.gradient(mean_total_loss, train_vars)
        optimizer.apply_gradients(zip(gradients, train_vars))

    # log to TensorBoard
    if writer is not None:
        prefix = 'train_' if is_train else 'validate_'
        with writer.as_default():
            classification_rate = tf.reduce_mean(
                tf.cast(tf.equal(prediction, labels), tf.float32))
            tf.summary.scalar(prefix + "classification_rate",
                              classification_rate,
                              step=global_step)
            tf.summary.scalar(prefix + "loss/mean classification",
                              tf.reduce_mean(classification_loss),
                              step=global_step)
            tf.summary.scalar(prefix +
                              "loss/mean input gradient regularization",
                              lambd * tf.reduce_mean(input_grad_reg_loss),
                              step=global_step)
            tf.summary.scalar(prefix + "loss/mean total loss",
                              mean_total_loss,
                              step=global_step)
    loss_numerator = tf.reduce_sum(classification_loss)
    accuracy_numerator = tf.reduce_sum(
        tf.cast(tf.equal(prediction, labels), tf.int32))
    denominator = x.shape[0]
    return loss_numerator, accuracy_numerator, denominator
