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
        """Calculates $E_\theta(x, a, w)$ where $x$ is data, $a$ is an
        attention mask on the data, and $w$ is a concept vector.

        Args:
            entity_model: model which calculates $g_\theta(x, w)$
            attention_model: model which calculates $f_\theta(g, w)$
        """
        super(ConceptEnergyModel, self).__init__(name=name)
        self._entity_model = entity_model
        self._attention_model = attention_model
        self._optimizer = tfp.optimizer.StochasticGradientLangevinDynamics(1e-4)

    def _build(self, data, attention, concept):
        """Calculates $E_\theta(x, a, w)$

        Args:


        TODO: doc"""
        return self._attention_model(
            self._entity_model(data, concept), attention, concept)

    def maximum_likelihood_loss(self, energy, attention_var):
        """Implements Eqn. 7"""
        attention_estimate = tf.random.normal(self._attention_shape)
        return -tf.nn.softplus(true_energy - attention_estimate_energy)

    def kl_loss(self):
        pass

    def stochastic_grad_langevin_dynamics_step(self, energy, variable, alpha,
                                               k):
        """Returns an operation which updates a variable with SGLD to minimize
        model energy"""
        grads_and_vars = self._optimizer.compute_gradients(
            energy, var_list=[variable])
        grad = grads_and_vars[0][0]
        omega = tf.random.normal(variable.shape, stddev=alpha)
        return self._optimizer.apply_gradients(
            [(variable + (alpha / 2.) * grad + omega**k, variable)])
