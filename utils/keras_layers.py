import tensorflow_probability as tfp

from tensorflow import layers
from tensorflow.keras import layers as k_layers
from tensorflow.keras import initializers
from utils.misc_utils import *
from utils.tf_utils import *

tfb = tfp.bijectors
tfd = tfp.distributions


class CondConvScaleShift(k_layers.Layer):

    def __init__(self, max_num_ratios, per_channel=True, normalize=True, max_scale_params=None, **kwargs):
        self.max_num_ratios = max_num_ratios
        self.per_channel = per_channel
        self.normalize = normalize
        if max_scale_params is not None:
            self.max_scales = tf_get_power_seq(*max_scale_params, n=max_num_ratios)
        else:
            self.max_scales = None

        super(CondConvScaleShift, self).__init__(**kwargs)

    def build(self, input_shape):

        WHC = input_shape[0][-3:]
        if self.per_channel:
            param_shape = [self.max_num_ratios, 1, 1, WHC.dims[-1].value]
        else:
            param_shape = [self.max_num_ratios, WHC.dims[0].value, WHC.dims[1].value, WHC.dims[2].value]

        self.alpha = self.add_weight(name='alpha',
                                     shape=param_shape,
                                     initializer=initializers.RandomNormal(mean=1.0, stddev=0.02),
                                     trainable=True)
        self.gamma = self.add_weight(name='gamma',
                                     shape=param_shape,
                                     initializer=initializers.RandomNormal(mean=1.0, stddev=0.02),
                                     trainable=True)
        self.beta = self.add_weight(name='beta',
                                    shape=param_shape,
                                    initializer=initializers.Zeros(),
                                    trainable=True)

        super(CondConvScaleShift, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x_y):
        x, y = x_y

        gamma = tf.gather(self.gamma, y, axis=0)  # (n_batch, ?, ?, num_channels)
        if self.max_scales is not None:
            max_scales = tf.gather(self.max_scales, y, axis=0)  # (n_batch, )
            max_scales = tf.reshape(max_scales, [-1, 1, 1, 1])  # (n_batch, 1, 1, 1)
            gamma = tf.clip_by_value(gamma, clip_value_min=-max_scales, clip_value_max=max_scales)

        beta = tf.gather(self.beta, y, axis=0)  # (n_batch, ?, ?, num_channels)

        if self.normalize:
            alpha = tf.gather(self.alpha, y, axis=0)  # (n_batch, ?, ?, num_channels)
            # normalize data across spatial locations
            means = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
            variances = tf.nn.moments(x, axes=[1, 2], keep_dims=True)[1]
            x = (x - means) / (tf.sqrt(variances + 1e-5))  # NWHC

            m = tf.reduce_mean(means, axis=-1, keepdims=True)
            v = tf.nn.moments(means, axes=[-1], keep_dims=True)[1]
            standardized_means = (means - m) / (tf.sqrt(v + 1e-5))

            x += standardized_means * alpha

        x *= gamma
        x += beta

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class CondScaleShift(k_layers.Layer):

    def __init__(self, max_num_ratios, dim_shape, name, max_scale_params, **kwargs):
        self.max_num_ratios = max_num_ratios
        self.size = dim_shape
        self.layer_name = name
        if max_scale_params is not None:
            self.max_scales = tf_get_power_seq(*max_scale_params, n=max_num_ratios)
        else:
            self.max_scales = None

        super(CondScaleShift, self).__init__(**kwargs)

    def build(self, input_shape):

        if isinstance(self.size, list):
            param_shape = (self.max_num_ratios, *self.size)
        else:
            param_shape = (self.max_num_ratios, self.size)

        self.scale_all = self.add_weight("{}_scale".format(self.layer_name),
                                       shape=param_shape,
                                       initializer=initializers.Ones(),
                                       trainable=True)
        self.bias_all = self.add_weight("{}_bias".format(self.layer_name),
                                      shape=param_shape,
                                      initializer=initializers.Zeros(),
                                      trainable=True)

        super(CondScaleShift, self).build(input_shape)

    def call(self, x_y):
        x, y = x_y

        scale = tf.gather(self.scale_all, y, axis=0)  # (?, hidden_size)
        if self.max_scales is not None:
            max_scales = tf.gather(self.max_scales, y, axis=0)  # (n_batch, )
            max_scales = tf.reshape(max_scales, [-1, 1])  # (n_batch, 1)
            scale = tf.clip_by_value(scale, clip_value_min=-max_scales, clip_value_max=max_scales)

        bias = tf.gather(self.bias_all, y, axis=0)  # (?, hidden_size)

        return scale * x + bias

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class CondDropout(k_layers.Layer):

    def __init__(self, start_rate, end_rate, rate_power, max_n_ratios, is_wmark_dim, **kwargs):
        print("dropout rates:")
        self.rates = tf_get_power_seq(start_rate, end_rate, rate_power, max_n_ratios)
        self.is_wmark_dim = is_wmark_dim
        super(CondDropout, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CondDropout, self).build(input_shape)

    def call(self, x_y, training=None):

        x, y = x_y
        if not training:
            return x

        x_shp = shape_list(x)  # (?, *event_dims) or (?, k, *event_dims) where k is num waymarks

        if self.is_wmark_dim:
            # the second dimension of x has size k, where k is the number of waymarks
            x_re = tf.reshape(x, x_shp[:2] + [-1])  # (?, k, d)

            # Sample a uniform distribution on [0.0, 1.0)
            random_tensor = tf.random.uniform(shape_list(x_re), dtype=x_re.dtype)  # (?, k, d)

            rates = tf.gather(self.rates, y, axis=0)  # (k, )
            scales = 1 / (1 - rates)  # (k, )

            keep_mask = tf.transpose(random_tensor, [0, 2, 1]) >= rates  # (? d, k)
            x_tr = tf.transpose(x_re, [0, 2, 1])  # (?, d, k)

            ret_tr = x_tr * scales * tf.cast(keep_mask, x.dtype)
            ret = tf.transpose(ret_tr, [0, 2, 1])  # (?, k, d)
            ret = tf.reshape(ret, x_shp)  # (?, k, *event_dims)

        else:
            # Sample a uniform distribution on [0.0, 1.0)
            random_tensor = tf.random.uniform(x_shp, dtype=x.dtype)  # (?, *event_dims)

            rates = tf.gather(self.rates, y, axis=0)  # (?, )
            event_ones = [1 for _ in x_shp[1:]]
            rates = tf.reshape(rates, [-1, *event_ones])  # (?, 1, ..., 1)

            keep_mask = random_tensor >= rates  # (?, *event_dims)
            scales = 1 / (1 - rates)  # (?, 1, ..., 1)
            ret = x * scales * tf.cast(keep_mask, x.dtype)  # (?, *event_dims)

        return ret

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class ScaleShift(k_layers.Layer):

    def __init__(self, size, name, use_shift=True, scale_init='ones', **kwargs):
        self.size = size
        self.layer_name = name
        self.use_shift = use_shift
        self.scale_init = scale_init
        super(ScaleShift, self).__init__(**kwargs)

    def build(self, input_shape):

        self.scale = self.add_weight("{}_scale".format(self.layer_name),
                                     shape=self.size,
                                     initializer=self.scale_init,
                                     trainable=True)
        if self.use_shift:
            self.shift = self.add_weight("{}_bias".format(self.layer_name),
                                         shape=self.size,
                                         initializer=initializers.Zeros(),
                                         trainable=True)

        super(ScaleShift, self).build(input_shape)

    def call(self, x):
        output = self.scale * x
        if self.use_shift: output += self.shift
        return output

    def compute_output_shape(self, input_shape):
        return input_shape


class SpectralConv(k_layers.Layer):

    def __init__(self,
                 n_channels,
                 kernel_shape,
                 strides=(1, 1),
                 padding='SAME',
                 name='conv',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 just_track_spectral_norm=False,
                 **kwargs):

        self.n_channels = n_channels
        self.kernel_shape = kernel_shape
        self.strides = strides
        self.padding = padding
        self.layer_name = name
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        # self.beta = max_spectral_norm
        self.just_track_spectral_norm = just_track_spectral_norm

        super(SpectralConv, self).__init__(**kwargs)

    def build(self, input_shape):
        print("building spectral normalisation conv layer")

        WHC  = input_shape[-3:]
        width, in_channels = WHC[0], WHC[-1]
        out_channels = self.n_channels

        self.w = self.add_weight("{}_kernel".format(self.layer_name),
                                 shape=[*self.kernel_shape, in_channels, out_channels],
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 trainable=True)

        self.u = self.add_weight("{}_power_iter_u".format(self.layer_name),
                                 shape=[1, width, width, in_channels],
                                 initializer=initializers.TruncatedNormal(0.0, 1.0),
                                 trainable=False)

        if self.use_bias:
            self.b = self.add_weight("{}_bias".format(self.layer_name),
                                     shape=self.n_channels,
                                     initializer=initializers.Zeros(),
                                     trainable=True)

        super(SpectralConv, self).build(input_shape)

    def call(self, x, training=True):
        w_norm = self.conv_spectral_norm(training)
        filter = self.w if self.just_track_spectral_norm else w_norm

        output = tf.nn.conv2d(input=x, filter=filter, strides=self.strides, padding='SAME')
        if self.use_bias:
            output += self.b
        return output

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], self.n_channels]

    def power_iteration_conv(self, num_iters=1):
        u = self.u
        u_shp = shape_list(u)  # u has shape [n_batch, width, height, n_channels]
        assert u_shp[1] == u_shp[2]

        u_ = u
        for _ in range(num_iters):
            v_ = l2_norm(tf.nn.conv2d(u_, self.w, padding='SAME'))
            u_ = l2_norm(tf.nn.conv2d_transpose(v_, self.w, [1, *u_shp[1:]], padding='SAME'))

        return u_, v_

    def conv_spectral_norm(self, training):

        u_hat, v_hat = self.power_iteration_conv()
        z = tf.nn.conv2d(u_hat, self.w, padding='SAME')

        sigma = tf.reduce_sum(tf.multiply(z, v_hat))
        # sigma = tf.maximum(sigma / self.beta, 1)

        if training:
            with tf.control_dependencies([self.u.assign(u_hat)]):
                w_norm = self.w / sigma
        else:
            w_norm = self.w / sigma

        self.spectral_norm = sigma

        return w_norm


class SpectralDense(k_layers.Layer):

    def __init__(self,
                 units,
                 name='dense',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 just_track_spectral_norm=False,
                 **kwargs):

        self.out_dim = units
        self.layer_name = name
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.just_track_spectral_norm = just_track_spectral_norm
        # self.beta = max_spectral_norm

        super(SpectralDense, self).__init__(**kwargs)

    def build(self, input_shape):
        print("building spectral normalisation dense layer")

        self.w = self.add_weight("{}_kernel".format(self.layer_name),
                                 shape=[input_shape[-1], self.out_dim],
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 trainable=True)

        self.u = self.add_weight("{}_power_iter_u".format(self.layer_name),
                                 shape=[1, self.out_dim],
                                 initializer=initializers.TruncatedNormal(0.0, 1.0),
                                 trainable=False)
        if self.use_bias:
            self.b = self.add_weight("{}_bias".format(self.layer_name),
                                     shape=self.out_dim,
                                     initializer=initializers.Zeros(),
                                     trainable=True)

        super(SpectralDense, self).build(input_shape)

    def call(self, x, training=True):
        w_norm = self.tf_spectral_norm(training)
        kernel = self.w if self.just_track_spectral_norm else w_norm
        output = tf.matmul(x, kernel)
        if self.use_bias: output += self.b
        return output

    def compute_output_shape(self, input_shape):
        return [*input_shape[:-1], self.out_dim]

    def power_iteration(self, num_iters=1):
        u_ = self.u
        for i in range(num_iters):
            # power iteration: usually iteration = 1 will be enough
            v_ = l2_norm(tf.matmul(u_, tf.transpose(self.w)))
            u_ = l2_norm(tf.matmul(v_, self.w))

        return u_, v_

    def tf_spectral_norm(self, training):

        u_hat, v_hat = self.power_iteration()
        sigma = tf.matmul(tf.matmul(v_hat, self.w), tf.transpose(u_hat))
        # sigma = tf.maximum(sigma / self.beta, 1)

        if training:
            with tf.control_dependencies([self.u.assign(u_hat)]):
                w_norm = self.w / sigma
        else:
            w_norm = self.w / sigma

        self.spectral_norm = sigma[0, 0]

        return w_norm


class KerasWeightMatrix(k_layers.Layer):

    def __init__(self,
                 in_dim,
                 out_dim=None,
                 activation=tf.identity,
                 init='zeros',
                 name='bias',
                 **kwargs):

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.shape = self.in_dim if self.out_dim is None else (self.in_dim, self.out_dim)

        self.init = init
        self.activation = activation
        self.layer_name = name
        super(KerasWeightMatrix, self).__init__(**kwargs)

    def build(self, _):

        self.w = self.add_weight(self.layer_name,
                                 shape=self.shape,
                                 initializer=self.init,
                                 trainable=True)

        super(KerasWeightMatrix, self).build(None)

    def call(self, _):
        return self.activation(self.w)

    def compute_output_shape(self, _):
        return self.shape


class KerasLTrilWeightMatrix(k_layers.Layer):

    def __init__(self,
                 dim,
                 name='LTril',
                 **kwargs):

        self.dim = dim
        self.layer_name = name
        super(KerasLTrilWeightMatrix, self).__init__(**kwargs)

    def build(self, _):

        self.W = self.add_weight(self.layer_name,
                                 shape=(self.dim, self.dim),
                                 initializer=initializers.zeros(),
                                 trainable=True)
        self.L = tf_enforce_lower_diag_and_nonneg_diag(self.W)

        super(KerasLTrilWeightMatrix, self).build(None)

    def call(self, _):
        return self.L

    def compute_output_shape(self, _):
        return [self.dim, self.dim]


class GatuOrTanh(k_layers.Layer):
    """Behaves like Gatu when input has WHC event dims, otherwise behaves like Tanh"""

    def __init__(self,
                 name="Gatu",
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 **kwargs):

        self.kernel_shape = [1, 1]
        self.layer_name = name
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

        super(GatuOrTanh, self).__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) > 3:
            WHC  = input_shape[-3:]
            width, in_channels = WHC[0], WHC[-1]
            out_channels = in_channels

            self.w1 = self.add_weight("{}_kernel1".format(self.layer_name),
                                     shape=[*self.kernel_shape, in_channels, out_channels],
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     trainable=True)

            self.w2 = self.add_weight("{}_kernel2".format(self.layer_name),
                                     shape=[*self.kernel_shape, in_channels, out_channels],
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     trainable=True)

        super(GatuOrTanh, self).build(input_shape)

    def call(self, x):
        if hasattr(self, "w1"):
            x_tanh = tf.nn.conv2d(input=x, filter=self.w1, padding='SAME')
            x_sigma = tf.nn.conv2d(input=x, filter=self.w2, padding='SAME')
            ret = tf.nn.tanh(x_tanh) * tf.nn.sigmoid(x_sigma)
        else:
            ret = tf.nn.tanh(x)

        return ret

    def compute_output_shape(self, input_shape):
        return input_shape


class ParallelDense(k_layers.Layer):
    """Computes multiple dense operations using a single matrix multiplication"""
    def __init__(self,
                 n_parallel,
                 in_dim,
                 out_dim,
                 activation=tf.identity,
                 init='glorot_uniform',
                 name='parallel_dense',
                 **kwargs):

        self.n_parallel =n_parallel
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.init = init
        self.activation = activation
        self.layer_name = name
        super(ParallelDense, self).__init__(**kwargs)

    def build(self, _):

        self.W = self.add_weight("W_{}".format(self.layer_name),
                                 shape=(self.n_parallel, self.out_dim, self.in_dim),
                                 initializer=self.init,
                                 trainable=True)
        self.b = self.add_weight("b_{}".format(self.layer_name),
                                 shape=(self.n_parallel, self.out_dim),
                                 initializer=self.init,
                                 trainable=True)

        super(ParallelDense, self).build(None)

    def call(self, x):
        """x has shape (batch_dim, in_dim)"""
        x_t = tf.transpose(x)
        Wx = tf.matmul(self.W, x_t)  # (n_parallel, out_dim, batch_dim)
        Wx = tf.transpose(Wx, [2, 0, 1])  # (batch_dim, n_parallel, out_dim)
        y = Wx + self.b
        return self.activation(y)

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.n_parallel, input_shape[1]]
