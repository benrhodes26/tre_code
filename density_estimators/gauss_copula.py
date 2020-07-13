import tensorflow_probability as tfp

from tensorflow.keras import initializers
from tensorflow.keras import layers as k_layers
from utils.keras_layers import *
from utils.tf_utils import *

tfb = tfp.bijectors
tfd = tfp.distributions


class GaussianCopulaFromSplines(tfd.TransformedDistribution):
    """
    Spline bijectors originate from https://github.com/bayesiains/nsf

    """
    def __init__(self, n_dims, event_reshape, num_splines=1,
                 spline_interval_min=-1, nbins_for_splines=128, per_dim_stds=None):

        self.init_x = tf.zeros(n_dims)
        self.n_dims = n_dims
        self.event_reshape = event_reshape
        self.per_dim_stds = per_dim_stds

        self.num_splines = num_splines
        self.spline_interval_min = spline_interval_min
        self.nbins = nbins_for_splines
        self.min_bin_size = self.min_slope = 1e-3

        self.base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(self.n_dims))  # standard normal

        super(GaussianCopulaFromSplines, self).__init__(
            distribution=self.base_dist,
            bijector=self.get_bijector(),
            validate_args=False,
            name="GaussianCopula")

    def get_bijector(self):

        bijectors = []

        self.cholesky = KerasLTrilWeightMatrix(self.n_dims, name="gauss_copula_cholesky")(self.init_x)
        self.mu = KerasWeightMatrix(self.n_dims, name="gauss_copula_mean")(self.init_x)
        self.cholesky_bijector = tfb.Affine(shift=self.mu, scale_tril=self.cholesky)
        bijectors.append(self.cholesky_bijector)

        aff1 = tfb.Affine(
            shift=KerasWeightMatrix(self.n_dims, init='zeros', name="affine_shift0")(self.init_x),
            scale_diag=KerasWeightMatrix((self.n_dims,), init='ones', name="affine_scale_diag0")(self.init_x)
        )
        rqs = tfb.RationalQuadraticSpline(
            bin_widths=KerasWeightMatrix(self.n_dims, self.nbins, activation=self._bin_positions, name='rqs_w_0')(self.init_x),
            bin_heights=KerasWeightMatrix(self.n_dims, self.nbins, activation=self._bin_positions, name='rqs_h_0')(self.init_x),
            knot_slopes=KerasWeightMatrix(self.n_dims, self.nbins - 1, activation=self._slopes, name='rqs_s_0')(self.init_x),
            range_min=self.spline_interval_min
        )
        aff2 = tfb.Affine(
            shift=KerasWeightMatrix(self.n_dims, init='zeros', name="affine_shift1")(self.init_x),
            scale_diag=KerasWeightMatrix((self.n_dims,), init='ones', name="affine_scale_diag1")(self.init_x)
        )

        self.marginal_bijector = tfb.Chain([aff2, rqs, aff1])
        bijectors.append(self.marginal_bijector)

        self.reshape_bijector = tfb.Reshape(event_shape_out=self.event_reshape)
        bijectors.append(self.reshape_bijector)

        if self.per_dim_stds is not None:
            self.std_rescale_bijector = tfb.Affine(scale_diag=tf.convert_to_tensor(self.per_dim_stds, dtype=tf.float32))
            bijectors.append(self.std_rescale_bijector)

        return tfb.Chain(list(reversed(bijectors)))

    def _bin_positions(self, w):
        interval_length = 2*np.abs(self.spline_interval_min)
        return tf.math.softmax(w, axis=-1) * (interval_length - (self.nbins * self.min_bin_size)) + self.min_bin_size

    def _slopes(self, w):
        const = np.log(np.exp(1 - self.min_slope) - 1)
        return tf.math.softplus(w + const) + self.min_slope

# if use_logit:
#     bijectors.append(tfb.Sigmoid())  # inverse maps data to [-13.8, 13.8]
#     bijectors.append(
#         tfb.Affine(
#             shift=-1e-6/(1 - 2e-6),
#             scale_identity_multiplier=1/(1 - 2e-6)  # inverse maps data to [1e-6, 1-1e-6]
#         )
#     )
#
#     shift = tf.cast(tf.reshape(shift, [-1]), tf.float32)
#     bijectors.append(tfb.Affine(shift=-shift))  # inverse uncentres the data
