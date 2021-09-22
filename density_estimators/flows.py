import tensorflow_probability as tfp

from density_estimators.mades import MogMade, residual_mog_made_template, residual_made_template
from density_estimators.gauss_copula import GaussianCopulaFromSplines
import numpy as KerasWeightMatrix
from utils.tf_utils import *

tfb = tfp.bijectors
tfd = tfp.distributions


class Flow:
    """Wrapper class to provide standard API for a variety of normalising flows"""

    def __init__(self,
                 input_dim,
                 num_bijectors,
                 n_layers_or_blocks,
                 hidden_size,
                 activation_name,
                 training,
                 n_mixture_components=10,
                 flow_type='GLOW',
                 use_batchnorm=False,
                 min_clip_norm=-2.,
                 max_clip_norm=2.,
                 clip_grad=False,
                 dropout_keep_p=1.,
                 reg_coef=0,
                 seed=None,
                 init_data = None,
                 img_shape = None,
                 glow_depth=8,
                 glow_use_split = True,
                 glow_coupling_type="rational_quadratic",
                 flow_num_spline_bins=8,
                 glow_temperature=1.0,
                 num_splines=1,
                 spline_interval_min=-1,
                 nbins_for_splines=128,
                 logit_copula_marginals=False,
                 data_minmax=None,
                 logit_alpha=None,
                 preprocess_shift=None,
                 preprocess_logit_shift=None,
                 per_dim_stds=None
                 ):
        self.input_dim = input_dim
        self.event_shape = shape_list(init_data)[1:]
        self.n_bijectors = num_bijectors
        self.n_layers_or_blocks = n_layers_or_blocks  # num layers for standard MLP, num blocks for residual MLP
        self.hidden_size = hidden_size
        self.activation_name = activation_name
        self.activation = get_tf_activation(activation_name)
        self.n_mixture_components = n_mixture_components  # only for MogMade
        self.training = training  # boolean to flag train or test
        self.flow_type = flow_type
        self.use_batchnorm = use_batchnorm
        self.min_clip_norm = min_clip_norm
        self.max_clip_norm = max_clip_norm
        self.clip_grad = clip_grad
        self.dropout_keep_p = dropout_keep_p
        self.reg_coef = reg_coef
        self.kernel_reg = tf_l2_regulariser(scale=self.reg_coef)

        self.seed = seed
        self.glow_depth = glow_depth if glow_depth else 8
        self.glow_use_split = glow_use_split
        self.glow_init_data = init_data  # need to initialise glow for stability
        self.glow_coupling_type = glow_coupling_type
        self.flow_num_spline_bins = flow_num_spline_bins
        self.glow_temperature = glow_temperature
        self.glow_init = None
        self.logit_alpha = logit_alpha
        self.preprocess_shift = preprocess_shift
        self.preprocess_logit_shift = preprocess_logit_shift

        self.num_splines = num_splines
        self.spline_interval_min = spline_interval_min
        self.nbins_for_splines = nbins_for_splines
        self.logit_copula_marginals = logit_copula_marginals
        self.per_dim_stds = per_dim_stds
        self.data_minmax = data_minmax

        self.img_shape = img_shape
        self.flow = self._make_flow()

    @property
    def base_dist(self):
        if hasattr(self.flow, "base_dist"):
            return self.flow.base_dist
        elif hasattr(self.flow, "distribution"):
            return self.flow.distribution
        else:
            raise ValueError

    @base_dist.setter
    def base_dist(self, value):
        if hasattr(self.flow, "base_dist"):
            self.flow.base_dist = value
        elif hasattr(self.flow, "distribution"):
            self.flow._distribution = value
        else:
            raise ValueError

    def _make_flow(self):
        """Build a normalising flow"""
        if self.flow_type in ['ResidualMAF', 'rq_nsf_coupling']:
            return self.make_flow()

        elif self.flow_type == 'GLOW':
            return self._make_glow()

        elif self.flow_type == 'GaussianCopula':
            # not typically referred to as a 'flow', but technically counts
            return self._make_gauss_copula()

        else:
            print("{} is not a valid choice of flow".format(self.flow_type))
            raise ValueError

    def make_flow(self):
        """Chain together multiple flow layers and return a TransformedDistribution"""

        base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(self.input_dim, tf.float32))
        bijectors = []
        for i in range(self.n_bijectors):

            if self.flow_type == 'ResidualMAF':
                bijectors.append(
                    tfb.MaskedAutoregressiveFlow(
                        shift_and_log_scale_fn=residual_made_template(
                            n_residual_blocks=self.n_layers_or_blocks,
                            hidden_units=self.hidden_size,
                            activation=self.activation(),
                            dropout_keep_p=self.dropout_keep_p,
                        )
                    )
                )
            elif self.flow_type == 'rq_nsf_coupling':
                half_d = int(self.input_dim / 2)
                bijectors.append(
                    tfb.RealNVP(num_masked=half_d,
                                bijector_fn=SplineBijectorFn(in_dims=half_d,
                                                             out_dims=half_d,
                                                             num_res_blocks=self.n_layers_or_blocks,
                                                             hidden_size=self.hidden_size,
                                                             nbins=self.flow_num_spline_bins
                                                             )
                                )
                )
            bijectors.append(
                tfb.Affine(
                    shift=KerasWeightMatrix(self.input_dim, init='zeros',
                                            name="affine_shift_{}".format(i))(tf.zeros(self.input_dim)),
                    scale_diag=KerasWeightMatrix((self.input_dim,), init='ones',
                                                 name="affine_scale_diag_{}".format(i))(tf.zeros(self.input_dim))
                )
            )

            # reverse the variable orderings
            bijectors.append(tfb.Permute(permutation=np.arange(self.input_dim)[::-1]))

        # Compose the bijectors and discard the last Permute layer
        self.bijector = tfb.Chain(list(reversed(bijectors[:-1])), name="chain_of_bijectors")
        dist = tfd.TransformedDistribution(distribution=base_dist, bijector=self.bijector, name="flow")
        return dist

    def _make_glow(self):
        from density_estimators.glow import Glow
        g = Glow(depth=self.glow_depth,
                 use_split=self.glow_use_split,
                 init_data=self.glow_init_data,
                 coupling_width=self.hidden_size,
                 coupling_type=self.glow_coupling_type,
                 num_spline_bins=self.flow_num_spline_bins,
                 img_shape=self.img_shape,
                 activation=self.activation_name,
                 temperature=self.glow_temperature,
                 dropout_rate=1.0-self.dropout_keep_p,
                 logit_alpha=self.logit_alpha,
                 shift=self.preprocess_shift,
                 logit_shift=self.preprocess_logit_shift
                 )
        self.glow_init = g.data_init
        return g

    def _make_gauss_copula(self):
        g = GaussianCopulaFromSplines(self.input_dim,
                                      self.event_shape,
                                      num_splines=self.num_splines,
                                      spline_interval_min=self.spline_interval_min,
                                      nbins_for_splines=self.nbins_for_splines,
                                      per_dim_stds=self.per_dim_stds
                                      )

        return g

    # def _make_mogmade(self):
    #     base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(self.input_dim, tf.float32))
    #     self.bijector = None  # mogmade is not actually a flow, so no bijector
    #     dist = MogMade(base_dist,
    #                    mog_param_fn=residual_mog_made_template(
    #                        n_out=3 * self.n_mixture_components,
    #                        n_residual_blocks=self.n_layers_or_blocks,
    #                        hidden_units=self.hidden_size,
    #                        activation=self.activation(),
    #                        dropout_keep_p=self.dropout_keep_p,
    #                    ),
    #                    n_dims=self.input_dim
    #                    )
    #     return dist

    def log_prob(self, x, *args, **kwargs):
        return self.flow.log_prob(x, *args, **kwargs)

    def sample(self, shape):
        return self.flow.sample(shape, seed=self.seed)

    def sample_base_dist(self, shape):
        return self.base_dist.sample(shape, seed=self.seed)

    # noinspection PyProtectedMember
    def inverse(self, x, ret_ildj=False):
        """encode data into z-space

        returns: z, log_det_jac"""

        if hasattr(self.flow, "inverse"):
            ret = self.flow.inverse(x) if ret_ildj else self.flow.inverse(x)[0]

        elif hasattr(self.flow, "bijector"):
            ret = self.flow.bijector.inverse(x)
            if ret_ildj:
                ildj = self.flow.bijector._inverse_log_det_jacobian(x)
                ret = (ret, ildj)
        else:
            print("self.flow does not implement an 'inverse' method, or have a"
                  "'bijector' attribute that implements it.")
            raise ValueError

        return ret

    # noinspection PyProtectedMember
    def forward(self, z, ret_ldj=False, collapse_wmark_dims=False):
        """decode data back into x-space

        returns: x, log_det_jac"""

        if collapse_wmark_dims:
            batch_shape = shape_list(z)[:2]
            z = tf.reshape(z, [-1, *shape_list(z)[2:]])

        if hasattr(self.flow, "forward"):
            ret = self.flow.forward(z) if ret_ldj else self.flow.forward(z)[0]

        elif hasattr(self.flow, "bijector"):
            ret = self.flow.bijector.forward(z)
            if ret_ldj:
                ldj = self.flow.bijector._forward_log_det_jacobian(z)
                ret = (ret, ldj)
        else:
            print("self.flow does not implement an 'forward' method, or have a"
                  "'bijector' attribute that implements it.")
            raise ValueError

        if collapse_wmark_dims:
            z = ret[0] if ret_ldj else ret
            z = tf.reshape(z, [*batch_shape, *shape_list(z)[1:]])
            ret = [z, ret[1]] if ret_ldj else z

        return ret


class SplineBijectorFn(tf.Module):

    def __init__(self, in_dims, out_dims, num_res_blocks, hidden_size, nbins=32, min_size=1e-3, spline_interval_min=-3):
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.num_blocks = num_res_blocks
        self.hidden_size = hidden_size
        self.nbins = nbins
        self.min_size = min_size
        self.spline_interval_min = spline_interval_min
        self.param_fn = self.build_param_fn()

    def _bin_positions(self, x):
        x = tf.reshape(x, [shape_list(x)[0], -1, self.nbins])
        interval_length = 2 * np.abs(self.spline_interval_min)
        return tf.math.softmax(x, axis=-1) * (interval_length - (self.nbins * self.min_size)) + self.min_size

    def _slopes(self, x):
        x = tf.reshape(x, [shape_list(x)[0], -1, self.nbins - 1])
        const = np.log(np.exp(1 - self.min_size) - 1)
        return tf.math.softplus(x + const) + self.min_size

    def build_param_fn(self):
        out_dims = self.out_dims
        output_shape = (3 * out_dims * self.nbins) - out_dims

        input_layer = tf.keras.layers.Dense(self.hidden_size, input_shape=(self.in_dims,))
        res_layers = [[tf.keras.layers.Dense(self.hidden_size, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)),
                       tf.keras.layers.Dense(self.hidden_size, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.1))]
                      for _ in range(self.num_blocks)]
        output_layer = tf.keras.layers.Dense(output_shape)

        def param_fn(x):
            res = self.residual_mlp(x, input_layer, res_layers, output_layer)
            w = self._bin_positions(res[:, :out_dims*self.nbins])
            h = self._bin_positions(res[:, out_dims*self.nbins : 2*out_dims*self.nbins])
            s = self._slopes(res[:, 2*out_dims*self.nbins:])
            return w, h, s

        return param_fn

    def residual_mlp(self, x, input_layer, res_layers, output_layer):
        h = input_layer(x)

        for i in range(self.num_blocks):
            residual = tf.keras.layers.LeakyReLU()(h)
            residual = res_layers[i][0](residual)
            residual = tf.keras.layers.LeakyReLU()(residual)
            # residual = Dropout(*self.dropout_params)(residual)
            residual = res_layers[i][1](residual)
            h += residual

        h = output_layer(h)
        return h

    def __call__(self, x, *args, **kwargs):
        w, h, s = self.param_fn(x)
        return tfb.RationalQuadraticSpline(bin_widths=w, bin_heights=h, knot_slopes=s)
