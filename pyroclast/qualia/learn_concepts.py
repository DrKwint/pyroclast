import tensorflow as tf

from pyroclast.qualia.concept_energy_model import ConceptEnergyModel
from pyroclast.qualia.models import build_attention_model, build_entity_model


def build_learn_from_demonstration(model, data_tensor, attention_tensor,
                                   concept_shape):
    concept_var = tf.get_variable(
        'concept_learning', dtype=tf.float32, shape=concept_shape)
    energy = model(data_tensor, attention_tensor, concept_var)
    k_placeholder = tf.placeholder(tf.float32, shape=())
    alpha_placeholder = tf.placeholder(tf.float32, shape=())
    w_update_op = model.stochastic_grad_langevin_dynamics_step(
        energy=energy,
        variable=concept_var,
        alpha=alpha_placeholder,
        k=k_placeholder)

    w_reset_op = concept_var.assign(tf.random.normal(shape=concept_var.shape))

    def learn_from_demonstration(session, sampling_iterations, alpha):
        session.run(w_reset_op)
        for k in range(sampling_iterations):
            session.run(w_update_op, {
                k_placeholder: float(k),
                alpha_placeholder: float(alpha)
            })

    return learn_from_demonstration


def learn(train_data,
          seed,
          entity_network='conv_only',
          attention_network='conv_only',
          sampling_iterations=10):
    """TODO: doc"""
    data_shape = train_data['image'].shape.as_list()[1:]
    if len(data_shape) == 3:
        # image data, make attention and concept a channel
        concept_shape = data_shape[:2] + [1]
        attention_shape = data_shape[:2] + [1]

    # setup graph
    entity_model_fn = build_entity_model(entity_network, strides=(1, 1, 1))
    attention_model_fn = build_attention_model(attention_network)
    model = ConceptEnergyModel(entity_model_fn, attention_model_fn)

    # build graph
    data_var = tf.get_variable(
        name='data', shape=[1] + data_shape, dtype=tf.float32)
    attention_var = tf.get_variable(
        name='attention', shape=[1] + attention_shape, dtype=tf.float32)
    learn_concept_fn = build_learn_from_demonstration(
        model, data_var, attention_var, concept_shape)

    concept_var = tf.get_variable(
        name='concept', shape=concept_shape, dtype=tf.float32)

    energy = model(data_var, attention_var, concept_var)

    return model
