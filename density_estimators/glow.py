# coding=utf-8

"""Glow generative model (https://arxiv.org/abs/1807.03039b)"""

from utils.misc_utils import *
from utils.tf_utils import *

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import warnings
from density_estimators import my_glow_ops as glow_ops

tfd = tfp.distributions
tfb = tfp.bijectors


# noinspection PyMethodOverriding
class Glow:
    """Glow generative model.

    This code is an adaptation of the Tensor2Tensor implementation.
    Vaswani, Ashish, et al. "Tensor2tensor for neural machine translation."
    arXiv preprint arXiv:1803.07416 (2018).

    For the original Glow paper: https://arxiv.org/abs/1807.03039"""

    def __init__(self,
                 depth,
                 use_split,
                 init_data,
                 coupling_width,
                 coupling_type,
                 num_spline_bins,
                 img_shape,
                 activation="relu",
                 temperature=1.0,
                 dropout_rate=0.0,
                 logit_alpha=None,
                 shift=None,
                 logit_shift=None,
                 ):

        if use_split:
            raise NotImplementedError("Currently, splitting off half the variables at every level"
                "will break. Would need to alter 'n_level' param, and shape logic in glow_ops.squeeze to enable it")

        # use residual param-outputting convnets with batchnorm?
        hparams = AttrDict({})

        # n_levels corresponds to the number of 'squeeze' operations we perform
        hparams.n_levels = 2 if img_shape[0] == 28 else 3

        # depth = number of revnet steps (Actnorm + invertible 1X1 conv + coupling) per level
        hparams.depth = depth
        hparams.use_split = use_split
        hparams.activation = activation  # relu or gatu
        hparams.coupling = coupling_type  # additive, affine or rational_quadratic
        hparams.num_spline_bins = num_spline_bins
        hparams.coupling_width = coupling_width
        hparams.coupling_dropout = dropout_rate
        hparams.clip_grad_norm = None

        # init_batch_size denotes the number of examples used for data-dependent
        # initialization. A higher init_batch_size is required for training
        # stability especially when hparams.batch_size is low.
        hparams.init_batch_size = 512
        hparams.img_shape = img_shape
        hparams.temperature = temperature
        hparams.logit_alpha = logit_alpha
        hparams.shift = shift
        hparams.logit_shift = logit_shift

        self.hparams = hparams
        self.n_dims = np.prod(np.array(img_shape))

        # create templates for parameter reuse
        self.encoder_decoder = self._encoder_decoder_template()
        self.base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(self.n_dims, dtype=tf.float32))

        # create data init operation
        self.data_init = self.log_prob(init_data, init=True)

    def log_prob(self, x, init=False):
        """
        Args:
          x: input data
          init: Whether or not to run data-dependent init.
        Returns:
          log_prob: float
        """
        x, batch_dims = self.collapse_batch_dims(x, is_latent=False)

        log_prob = 0.
        self.z, encoder_ildj = self.encoder_decoder(x, reverse=False, init=init, temperature=self.hparams.temperature)
        log_prob += encoder_ildj  # inverse log-det jacobian
        log_prob += self.base_dist.log_prob(self.z)  # base dist over latent z

        log_prob = self.expand_batch_dims(log_prob, batch_dims)
        return log_prob

    def inverse(self, x):
        """Given observable x, return the latent z (and inverse log det jacobian)"""
        x, batch_dims = self.collapse_batch_dims(x, is_latent=False)
        z, ildj = self.encoder_decoder(x, reverse=False)

        z = self.expand_batch_dims(z, batch_dims)
        ildj = self.expand_batch_dims(ildj, batch_dims)
        return z, ildj

    def forward(self, z):
        """Given latent z, return the observable x (and log det jacobian)"""
        z, batch_dims = self.collapse_batch_dims(z, is_latent=True)
        x, ldj = self.encoder_decoder(z, reverse=True, temperature=self.hparams.temperature)

        x = self.expand_batch_dims(x, batch_dims)
        ldj = self.expand_batch_dims(ldj, batch_dims)
        return x, ldj

    def sample_prior(self, n_sample, seed):
        """Sample from the latent z space"""
        z_sample = self.base_dist.sample(n_sample, seed=seed)
        return z_sample

    def sample(self, n_sample, seed):
        """Sample from the observable x space"""
        z_sample = self.sample_prior(n_sample, seed)
        x_sample = self.forward(z_sample)[0]
        return x_sample

    def _encoder_decoder_template(self, name=None):
        name = name or "glow_encoder_decoder_template"

        def _fn(u, reverse, init=False, temperature=1.0):
            results = glow_ops.encoder_decoder("codec", u, self.hparams, reverse=reverse, init=init, temperature=temperature)
            return results

        return tf.make_template(name, _fn)

    @staticmethod
    def collapse_batch_dims(u, is_latent):
        u_shp = shape_list(u)
        batch_dims = u_shp[:1]  # by default, first dim is batch dim & the rest are event dims

        if is_latent and len(u_shp) > 2:
            u = tf.reshape(u, [-1, u_shp[-1]])
            batch_dims = u_shp[:-1]

        elif (not is_latent) and len(u_shp) > 4:
            u = tf.reshape(u, [-1, *u_shp[-3:]])
            batch_dims = u_shp[:-3]

        return u, batch_dims

    @staticmethod
    def expand_batch_dims(x, batch_dims):
        x_shp = shape_list(x)
        return tf.reshape(x, batch_dims + x_shp[1:])
