import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp


class M2VAE(snt.AbstractModule):
    """M2 VAE model from Kinga et al. 2014 'Semi-supervised Learning with Deep
    Generative Models'
    """

    def __init__(self,
                 classifier,
                 encoder,
                 decoder,
                 prior,
                 posterior,
                 output_dist,
                 name='m2_vae'):
        """
        Args:
            classifier (Tensor -> Tensor): Called $q_\phi(y|x)$, takes data and
                gives class logits
            encoder (Tensor -> Tensor -> (Tensor, Tensor)): Called
                $q_\phi(z|x,y)$, takes data and class label and outputs loc and
                scale values
            decoder (Tensor -> Tensor): Called $p_\theta(x|y,z)$, takes
                class label and latent vector and outputs data reconstruction
            prior (tfp.Distribution): distribution with sample shape equal to latent shape
            posterior (Tensor -> Tensor -> tfp.Distribution): Callable which
                takes location and scale and returns a tfp distribution
            output_dist (Tensor -> tfp.Distribution): Callable from loc to a
                tfp distribution
        """
        super(M2VAE, self).__init__(name=name)

        # TODO: put these asserts in the right places
        # assert prior sample shape == latent shape
        # assert loc shape == latent shape
        # assert scale shape == latent shape
        # assert posterior sample shape == latent shape

        self._classifier = classifier
        self._encoder = encoder
        self._decoder = decoder
        self._prior = prior
        self._posterior = posterior
        self._output_dist = output_dist

        # TODO: add public distribution to sample (x,y,z) from this model

    def _build(self, inputs):
        """
        Args:
            inputs (Tensor, Tensor): split into x and temperature, the first is
                input data and the second is the temperature for the
                RelaxedOneHotCategorical distribution used in the inferring
                `p_y`. `temperature` should be annealed from 1 to 0 during
                training.

        Returns:
            (tfp.Distribution, tfp.Distribution, tfp.Distribution): output
                distribution `p_x`, classification `p_y`, and latent posterior
                `p_z`
        """
        x, temperature = inputs

        # calculate $q_\phi(y|x)$ and sample `y_hat`
        class_logits = self._classifier(x)
        p_y = tfp.distributions.RelaxedOneHotCategorical(
            temperature, logits=class_logits)
        y_hat = p_y.sample()

        # calculate $q_\phi(z|x,y)$ and sample `z`
        loc, scale = self._encoder(x, y_hat)
        p_z = self._posterior(loc, scale)
        z = p_z.sample()

        # calculate $p_\theta(x|y,z)$
        output_scale = self._decoder(y_hat, z)
        p_x = tfp.distributions.Independent(
            self._output_dist(output_scale), reinterpreted_batch_ndims=2)

        return p_x, p_y, p_z

    def supervised_loss(self, x, p_x, p_z):
        """Calculate $-\mathcal{L}(x,y)$

        Args:
            x (Tensor): data
            p_x (tfp.Distribution): model output distribution
            p_z (tfp.Distribution): model latent posterior distribution

        Returns:
            Tensor: scalar loss value
        """
        logpx = p_x.log_prob(x)
        kl = tfp.distributions.kl_divergence(p_z, self._prior)
        return -(logpx - kl)

    def unsupervised_loss(self, x, p_x, p_y, p_z):
        """Calculate $-\mathcal{U}{x}

        Args:
            x (Tensor):
            y (Tensor):
            p_x (Tensor):
            p_y (Tensor):
            p_z (Tensor):
            # TODO: finish these docs

        Returns:
            Tensor: scalar loss value
        """
        y_values = tf.range(tf.shape(p_y)[-1])
        supervised_component = self.supervised_loss(x, p_x, p_z)
        logqy = tfp.distributions.Categorical(
            logits=p_y.logits).log_prob(y_values)
        # TODO: figure out shapes here
        print(supervised_component.shape)
        print(logqy.shape)
        # TODO: write the rest of this function
        return 0.

    def loss(self, labeled_x, unlabeled_x, y, model_y, model_z, model_x):
        """
        # TODO: document
        """
        # TODO: write this as J^alpha from the paper
        pass
