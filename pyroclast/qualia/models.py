import sonnet as snt
import tensorflow as tf


def build_entity_model(network, **network_kwargs):
    if isinstance(network, str):
        from pyroclast.common.models import get_network_builder
        network = get_network_builder(network)(**network_kwargs)

    def entity_model_builder(data, concept):
        broadcast_concept = tf.broadcast_to(concept, data.shape)
        broadcast_concept = tf.dtypes.cast(broadcast_concept, tf.float32)
        network_input = tf.concat([data, broadcast_concept], axis=-1)
        return network(network_input)

    return entity_model_builder


def build_attention_model(network, **network_kwargs):
    if isinstance(network, str):
        from pyroclast.common.models import get_network_builder
        network = get_network_builder(network)(**network_kwargs)

    def attention_model_builder(data, attention, concept):
        attended_data = tf.nn.sigmoid(attention) * data
        broadcast_concept = tf.broadcast_to(concept, attended_data.shape)
        broadcast_concept = tf.dtypes.cast(broadcast_concept, tf.float32)
        network_input = tf.concat([attended_data, broadcast_concept], axis=-1)
        return tf.square(network(network_input))

    return attention_model_builder
