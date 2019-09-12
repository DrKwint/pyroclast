import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from pyroclast.cpvae.ddt import transductive_box_inference, get_decision_tree_boundaries
import sklearn.tree


class CpVAE(tf.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 decision_tree,
                 img_height,
                 img_width,
                 latent_dimension,
                 class_num,
                 box_num,
                 name='cpvae'):
        super(CpVAE, self).__init__(name=name)
        self.encoder = encoder
        self.decoder = decoder
        self.decision_tree = decision_tree
        self.img_height = img_height
        self.img_width = img_width
        self.z_prior = None

        # tree_stuff
        self._lower = tf.Variable(
            np.empty([box_num, latent_dimension], dtype=np.float32),
            trainable=False)
        self._upper = tf.Variable(
            np.empty([box_num, latent_dimension], dtype=np.float32),
            trainable=False)
        self._values = tf.Variable(
            np.empty([box_num, class_num], dtype=np.float32), trainable=False)

    def __call__(self, x):
        loc, scale_diag = self._encode(x)
        z_posterior = tfp.distributions.MultivariateNormalDiag(
            loc=loc, scale_diag=tf.nn.softplus(scale_diag))
        z = z_posterior.sample()
        x_hat = self._decode(z)
        y_hat = transductive_box_inference(loc, scale_diag, self._lower,
                                             self._upper, self._values)

        if self.z_prior is None:
            self.z_prior = tfp.distributions.MultivariateNormalDiag(
                tf.zeros(z_posterior.event_shape),
                tf.ones(z_posterior.event_shape))
        return x_hat, y_hat, z_posterior

    def _encode(self, x):
        loc, scale_diag = self.encoder(x)
        return loc, scale_diag

    def _decode(self, z):
        return tf.image.resize_with_crop_or_pad(
            self.decoder(z), self.img_height, self.img_width)

    def sample(self, sample_shape=(1), z=None):
        if z is None:
            z = self.z_prior.sample(sample_shape)
        return self._decode(z)

    def vae_loss(self, x, x_hat, z_posterior):
        distortion = tf.losses.mean_squared_error(labels=x, predictions=x_hat)
        rate = tfp.distributions.kl_divergence(z_posterior, self.z_prior)
        return distortion, rate

    def calculate_latent_stats(self, dataset):
        # run data
        labels = dataset.map(lambda x: x['label'])
        loc_scale_pairs = dataset.map(lambda x: self._encode(x['image']))
        z_samples = loc_scale_pairs.map(lambda x: tfp.distributions.MultivariateNormalDiag(loc=x[0], scale_diag=tf.nn.softplus(x[1])).sample())
        
        
        codes = []
        labels = []
        mu = []
        sigma = []
        for _ in range(batch_num):
            loc, scale_diag = self._encode(x)
            
            z = z_posterior.sample()
            c, m, s, l = session.run(
                [
                    self.latent_posterior_sample, self.z_mu, self.z_sigma,
                    label_tensor
                ],
                feed_dict=feed_dict_fn())
            codes.append(c)
            labels.append(l)
            mu.append(m)
            sigma.append(s)
        mu = np.concatenate(mu)
        sigma = np.concatenate(sigma)
        codes = np.squeeze(np.concatenate(codes, axis=1))
        labels = np.argmax(np.concatenate(labels), axis=1)
        sigma = np.array(sigma)
        return mu, sigma, codes, labels

"""
    def calculate_latent_params_by_class(loc, scale_diag, class_num, latent_dimension):
        # update class stats
        if len(labels.shape) > 1: labels = np.argmax(labels, axis=1)
        class_locs = np.empty([class_num, latent_dimension])
        class_scales = np.empty([class_num, latent_dimension])
        loc_sq = np.square(loc)
        scale_sq = np.square(scale)
        sum_sq = scale_sq + loc_sq
        for l in range(class_num):
            idxs = np.nonzero(labels == l)[0]
            class_locs[l] = np.mean(loc[idxs], axis=0)
            class_scales[l] = np.mean(
                sum_sq[idxs], axis=0) - np.square(class_locs[l])
        return class_loc, class_scale_diag

    def calculate_dt_boxes(decision_tree, z, label, class_num, latent_dimension):
        # train ensemble
        decision_tree.fit(codes, labels)
        lower_, upper_, values_ = ddt.get_decision_tree_boundaries(
            decision_tree, latent_dimension, class_num)
        # ensure arrays are of correct size, even if tree is too small
        lower[:lower_.shape[0], :lower_.shape[1]] = lower_
        upper[:upper_.shape[0], :upper_.shape[1]] = upper_
        values[:values_.shape[0], :values_.shape[1]] = values_
        return lower, upper, values

    tree.export_graphviz(
        self._decision_tree,
        out_file=os.path.join(output_dir, 'ddt_epoch{}.dot'.format(epoch)),
        filled=True,
        rounded=True)
"""