"""Implementation of [Concept Learning with Energy-Based Models](https://arxiv.org/pdf/1811.02486.pdf)"""

import tensorflow as tf
import sonnet as snt
import tensorflow_probability as tfp


class ConceptEnergyModel(snt.AbstractModule):
    """A model which, given data $x$, attention mask $a$, and concept vector
    $w$, calculates the energy function. This should be zero if the attention
    mask

    TODO: doc
    """

    def __init__(self,
                 entity_model,
                 attention_model,
                 name='concept_energy_model'):
        """TODO: doc

        Args:
            entity_model:
            attention_model:
        """
        super(ConceptEnergyModel, self).__init__(name=name)
        self._entity_model = entity_model
        self._attention_model = attention_model

    def _build(self, data, attention, concept):
        """Calculates $E_\theta(x, a, w)$

        Args:


        TODO: doc"""
        g_theta = self._entity_model(data, concept)
        e_theta = tf.square(
            self._attention_model(tf.nn.sigmoid(attention) * g_theta, concept))

        return e_theta

    def optimize_attention(self, ):
        """TODO: doc"""
        optimizer = tfp.optimizer.StochasticGradientLangevinDynamics(
            learning_rate=0.01)
