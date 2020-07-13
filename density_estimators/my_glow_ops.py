# The code below is adapted from public Tensor2Tensor code
# FYI, another decent repo (using keras) is https://github.com/bojone/flow

# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Various reversible ops for the glow generative model."""

from utils.tf_utils import *
from utils.misc_utils import logit, sigmoid
import functools
import numpy as np
import scipy

import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors


class TemperedNormal(tfp.distributions.Normal):
    """Normal distribution with temperature T, used as base distribution over latent z"""

    def __init__(self, loc, scale, temperature=1.0):
        self.temperature = temperature
        new_scale = scale * self.temperature
        tfp.distributions.Normal.__init__(self, loc=loc, scale=new_scale, name="TemperedNormal")

    def log_prob(self, z):
        lp = super(TemperedNormal, self).log_prob(z)
        return tf.reduce_sum(lp, axis=[1, 2, 3])  # sum over height, width, channels

    def sample(self, sample_shape, seed=None, name="sample"):
        z = super(TemperedNormal, self).sample(sample_shape=sample_shape, seed=seed, name=name)
        return z


def default_initializer(std = 0.05):
    return tf.random_normal_initializer(0., std)


def get_eps(dist, x):
    """Z = (X - mu) / sigma."""
    return (x - dist.loc) / dist.scale


def set_eps(dist, eps):
    """Z = eps * sigma + mu."""
    return eps * dist.scale + dist.loc


def assign(w, initial_value):
    w = w.assign(initial_value)
    with tf.control_dependencies([w]):
        return w


def get_variable_ddi(name, shape, initial_value, dtype = tf.float32, init = False, trainable = True):
    """Wrapper for data-dependent initialization."""
    # If init is a tensor bool, w is returned dynamically.
    w = tf.get_variable(name, shape, dtype, None, trainable=trainable)
    if isinstance(init, bool):
        if init:
            return assign(w, initial_value)
        return w
    else:
        return tf.cond(init, lambda: assign(w, initial_value), lambda: w)


def get_dropout(x, rate = 0.0, init = False):
    """Zero dropout during init or prediction time.

    Args:
      x: 4-D Tensor, shape=(NHWC).
      rate: Dropout rate.
      init: Initialization.
    Returns:
      x: activations after dropout.
    """
    if init or rate == 0:
        return x
    return tf.layers.dropout(x, rate=rate, training=True)


def actnorm(name, x, logscale_factor = 1., reverse = False, init = False):
    """x_{ij} = s x x_{ij} + b. Per-channel scaling and bias.

    If init is set to True, the scaling and bias are initialized such
    that the mean and variance of the output activations of the first minibatch
    are zero and one respectively.

    Args:
      name: variable scope.
      x: input
      logscale_factor: Used in actnorm_scale. Optimizes f(ls*s') instead of f(s)
                       where s' = s / ls. Helps in faster convergence.
      reverse: forward or reverse operation.
      init: Whether or not to do data-dependent initialization.
      trainable:

    Returns:
      x: output after adding bias and scaling.
      objective: log(sum(s))
    """
    var_scope = tf.variable_scope(name, reuse=tf.AUTO_REUSE)

    with var_scope:
        if not reverse:
            x = actnorm_center(name + "_center", x, reverse, init=init)
            x, objective = actnorm_scale(
                name + "_scale", x, logscale_factor=logscale_factor, reverse=reverse, init=init)
        else:
            x, objective = actnorm_scale(
                name + "_scale", x, logscale_factor=logscale_factor, reverse=reverse, init=init)
            x = actnorm_center(name + "_center", x, reverse, init=init)

        return x, objective


def actnorm_center(name, x, reverse = False, init = False):
    """Add a bias to x.

    Initialize such that the output of the first minibatch is zero centered
    per channel.

    Args:
      name: scope
      x: 2-D or 4-D Tensor.
      reverse: Forward or backward operation.
      init: data-dependent initialization.

    Returns:
      x_center: (x + b), if reverse is True and (x - b) otherwise.
    """
    shape = shape_list(x)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        assert len(shape) == 2 or len(shape) == 4

        if len(shape) == 2:
            x_mean = tf.reduce_mean(x, [0], keepdims=True)
            b = get_variable_ddi("b", (1, shape[1]), initial_value=-x_mean, init=init)

        elif len(shape) == 4:
            x_mean = tf.reduce_mean(x, [0, 1, 2], keepdims=True)
            b = get_variable_ddi("b", (1, 1, 1, shape[3]), initial_value=-x_mean, init=init)

        if not reverse:
            x += b
        else:
            x -= b
        return x


def actnorm_scale(name, x, logscale_factor = 1., reverse = False, init = False):
    """Per-channel scaling of x."""
    x_shape = shape_list(x)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        # Variance initialization logic.
        assert len(x_shape) == 2 or len(x_shape) == 4
        if len(x_shape) == 2:
            x_var = tf.reduce_mean(x ** 2, [0], keepdims=True)
            logdet_factor = 1
            var_shape = (1, x_shape[1])
        elif len(x_shape) == 4:
            x_var = tf.reduce_mean(x ** 2, [0, 1, 2], keepdims=True)
            logdet_factor = x_shape[1] * x_shape[2]
            var_shape = (1, 1, 1, x_shape[3])

        init_value = tf.log(1.0 / (tf.sqrt(x_var) + 1e-6)) / logscale_factor
        logs = get_variable_ddi("logs", var_shape, initial_value=init_value, init=init)
        logs = logs * logscale_factor

        # Function and reverse function.
        if not reverse:
            x = x * tf.exp(logs)
        else:
            x = x * tf.exp(-logs)

        # Objective calculation, h * w * sum(log|s|)
        dlogdet = tf.reduce_sum(logs) * logdet_factor
        if reverse:
            dlogdet *= -1

        return x, dlogdet


def invertible_1x1_conv(name, x, reverse = False):
    """1X1 convolution on x.

    The 1X1 convolution is parametrized as P*L*(U + sign(s)*exp(log(s))) where
    1. P is a permutation matrix.
    2. L is a lower triangular matrix with diagonal entries unity.
    3. U is a upper triangular matrix where the diagonal entries zero.
    4. s is a vector.

    sign(s) and P are fixed and the remaining are optimized. P, L, U and s are
    initialized by the PLU decomposition of a random rotation matrix.

    Args:
      name: scope
      x: Input Tensor.
      reverse: whether the pass is from z -> x or x -> z.

    Returns:
      x_conv: x after a 1X1 convolution is applied on x.
      objective: sum(log(s))
    """
    _, height, width, channels = shape_list(x)
    w_shape = [channels, channels]

    # Random rotation-matrix Q
    random_matrix = np.random.rand(channels, channels)
    np_w = scipy.linalg.qr(random_matrix)[0].astype("float32")

    # Initialize P,L,U and s from the LU decomposition of a random rotation matrix
    np_p, np_l, np_u = scipy.linalg.lu(np_w)
    np_s = np.diag(np_u)
    np_sign_s = np.sign(np_s)
    np_log_s = np.log(np.abs(np_s))
    np_u = np.triu(np_u, k=1)

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        p = tf.get_variable("P", initializer=np_p, trainable=False)
        l = tf.get_variable("L", initializer=np_l)
        sign_s = tf.get_variable("sign_S", initializer=np_sign_s, trainable=False)
        log_s = tf.get_variable("log_S", initializer=np_log_s)
        u = tf.get_variable("U", initializer=np_u)

        # W = P * L * (U + sign_s * exp(log_s))
        l_mask = np.tril(np.ones([channels, channels], dtype=np.float32), -1)
        l = l * l_mask + tf.eye(channels, channels)
        u = u * np.transpose(l_mask) + tf.diag(sign_s * tf.exp(log_s))
        w = tf.matmul(p, tf.matmul(l, u))

        # If height or width cannot be statically determined then they end up as
        # tf.int32 tensors, which cannot be directly multiplied with a floating
        # point tensor without a cast.
        objective = tf.reduce_sum(log_s) * tf.cast(height * width, log_s.dtype)
        if not reverse:
            w = tf.reshape(w, [1, 1] + w_shape)
            x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format="NHWC")
        else:
            u_inv = tf.matrix_inverse(u)
            l_inv = tf.matrix_inverse(l)
            p_inv = tf.matrix_inverse(p)
            w_inv = tf.matmul(u_inv, tf.matmul(l_inv, p_inv))
            w_inv = tf.reshape(w_inv, [1, 1] + w_shape)
            x = tf.nn.conv2d(x, w_inv, [1, 1, 1, 1], "SAME", data_format="NHWC")
            objective *= -1

    return x, objective


def add_edge_bias(x, filter_size):
    """Pad x and concatenates an edge bias across the depth of x.

    The edge bias can be thought of as a binary feature which is unity when
    the filter is being convolved over an edge and zero otherwise.

    Args:
      x: Input tensor, shape (NHWC)
      filter_size: filter_size to determine padding.
    Returns:
      x_pad: Input tensor, shape (NHW(c+1))
    """
    x_shape = shape_list(x)
    if filter_size[0] == 1 and filter_size[1] == 1:
        return x
    a = (filter_size[0] - 1) // 2  # vertical padding size
    b = (filter_size[1] - 1) // 2  # horizontal padding size
    padding = [[0, 0], [a, a], [b, b], [0, 0]]
    x_bias = tf.zeros(x_shape[:-1] + [1])

    x = tf.pad(x, padding)
    x_pad = tf.pad(x_bias, padding, constant_values=1)
    return tf.concat([x, x_pad], axis=3)


def conv(name,
         x,
         output_channels,
         filter_size = None,
         stride = None,
         logscale_factor = 1.0,
         apply_actnorm = True,
         conv_init = "default",
         dd_init=False,
         dilations = None):
    """Convolutional layer with edge bias padding and optional actnorm.

    Args:
      name: variable scope.
      x: 4-D Tensor Tensor of shape NHWC
      output_channels: Number of output channels.
      filter_size: list of ints, if None [3, 3] is the default
      stride: list of ints, default stride: 1
      logscale_factor: see actnorm for parameter meaning.
      apply_actnorm: if apply_actnorm the activations of the first minibatch
                     have zero mean and unit variance. Else, there is no scaling
                     applied.
      conv_init: default or zeros. default is a normal distribution with 0.05 std.
      dilations: List of integers, apply dilations.
    Returns:
      x: actnorm(conv2d(x))
    Raises:
      ValueError: if init is set to "zeros" and apply_actnorm is set to True.
    """
    if conv_init == "zeros" and apply_actnorm:
        raise ValueError("apply_actnorm is unstable when init is set to zeros.")

    # set filter_size, stride and in_channels
    if filter_size is None:
        filter_size = [3, 3]
    if stride is None:
        stride = [1, 1]
    if dilations is None:
        dilations = [1, 1, 1, 1]

    x = add_edge_bias(x, filter_size=filter_size)

    in_channels = shape_list(x)[-1]
    filter_shape = filter_size + [in_channels, output_channels]
    stride_shape = [1] + stride + [1]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        if conv_init == "default":
            initializer = default_initializer()
        elif conv_init == "zeros":
            initializer = tf.zeros_initializer()

        w = tf.get_variable("W", filter_shape, tf.float32, initializer=initializer)
        x = tf.nn.conv2d(x, w, stride_shape, padding="VALID", dilations=dilations)
        if apply_actnorm:
            x, _ = actnorm("actnorm", x, logscale_factor=logscale_factor, init=dd_init)
        else:
            x += tf.get_variable("b", [1, 1, 1, output_channels],
                                 initializer=tf.zeros_initializer())
            logs = tf.get_variable("logs", [1, output_channels],
                                   initializer=tf.zeros_initializer())
            x *= tf.exp(logs * logscale_factor)
        return x


def conv_block(name, x, mid_channels, dilations = None, activation = "relu", dd_init=False, dropout = 0.0):
    """2 layer conv block used in the affine coupling layer.

    Args:
      name: variable scope.
      x: 4-D Tensor.
      mid_channels: Output channels of the second layer.
      dilations: Optional, list of integers.
      activation: relu or gatu.
        If relu, the second layer is relu(W*x)
        If gatu, the second layer is tanh(W1*x) * sigmoid(W2*x)
      dropout: Dropout probability.
    Returns:
      x: 4-D Tensor: Output activations.
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        # Edge Padding + conv2d + actnorm + relu:
        # [output: 512 channels]
        x = conv("1_1", x, output_channels=mid_channels, filter_size=[3, 3], dilations=dilations, dd_init=dd_init)
        x = tf.nn.relu(x)
        x = get_dropout(x, rate=dropout, init=dd_init)

        # Padding + conv2d + actnorm + activation.
        # [input, output: 512 channels]
        if activation == "relu":
            x = conv("1_2", x, output_channels=mid_channels, filter_size=[1, 1], dilations=dilations, dd_init=dd_init)
            x = tf.nn.relu(x)
        elif activation == "gatu":
            # x = tanh(w1*x) * sigm(w2*x)
            x_tanh = conv("1_tanh", x, output_channels=mid_channels, filter_size=[1, 1], dilations=dilations, dd_init=dd_init)
            x_sigm = conv("1_sigm", x, output_channels=mid_channels, filter_size=[1, 1], dilations=dilations, dd_init=dd_init)
            x = tf.nn.tanh(x_tanh) * tf.nn.sigmoid(x_sigm)

        x = get_dropout(x, rate=dropout, init=dd_init)
        return x


def conv_stack(name, x, mid_channels, output_channels, dilations = None, activation = "relu", dd_init=False, dropout = 0.0):
    """3-layer convolutional stack.

    Args:
      name: variable scope.
      x: 4-D Tensor.
      mid_channels: Number of output channels of the first layer.
      output_channels: Number of output channels.
      dilations: Dilations to apply in the first 3x3 layer and the last 3x3 layer. By default, apply no dilations.
      activation: relu or gatu.
        If relu, the second layer is relu(W*x)
        If gatu, the second layer is tanh(W1*x) * sigmoid(W2*x)
      dropout: float, 0.0
    Returns:
      output: output of 3 layer conv network.
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x = conv_block("conv_block",
                       x,
                       mid_channels=mid_channels,
                       dilations=dilations,
                       activation=activation,
                       dd_init=dd_init,
                       dropout=dropout)
        # Final layer.
        x = conv("zeros", x, apply_actnorm=False, conv_init="zeros",
                 output_channels=output_channels, dd_init=dd_init, dilations=dilations)
    return x


def additive_coupling(name, x, mid_channels = 512, reverse = False, activation = "relu", dd_init=False, dropout = 0.0):
    """Reversible additive coupling layer.

    Args:
      name: variable scope.
      x: 4-D Tensor.
      mid_channels: number of channels in the coupling layer.
      reverse: Forward or reverse operation.
      activation: "relu" or "gatu"
      dropout: default, 0.0
    Returns:
      output:
      objective: 0.0
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        output_channels = shape_list(x)[-1] // 2
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)

        z1 = x1
        shift = conv_stack("nn", x1, mid_channels, output_channels=output_channels,
                           activation=activation, dd_init=dd_init, dropout=dropout)

        if not reverse:
            z2 = x2 + shift
        else:
            z2 = x2 - shift
        return tf.concat([z1, z2], axis=3), 0.0



def affine_coupling(name, x, mid_channels = 512, activation = "relu", reverse = False, dd_init=False, dropout = 0.0):
    """Reversible affine coupling layer.

    Args:
      name: variable scope.
      x: 4-D Tensor.
      mid_channels: number of channels in the coupling layer.
      activation: Can be either "relu" or "gatu".
      reverse: Forward or reverse operation.
      dropout: default, 0.0
    Returns:
      output: input s
      objective: log-determinant of the jacobian
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x_shape = shape_list(x)
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)

        # scale, shift = NN(x1)
        # If reverse: z2 = scale * (x2 + shift)
        # Else: z2 = (x2 / scale) - shift
        z1 = x1
        log_scale_and_shift = conv_stack(
            "nn", x1, mid_channels, x_shape[-1], activation=activation, dd_init=dd_init, dropout=dropout)

        shift = log_scale_and_shift[:, :, :, 0::2]
        scale = 1e-3 + tf.nn.softplus(log_scale_and_shift[:, :, :, 1::2] + np.log(np.e - 1))

        if not reverse:
            z2 = (x2 + shift) * scale
        else:
            z2 = x2 / scale - shift

        objective = tf.reduce_sum(tf.log(scale), axis=[1, 2, 3])
        if reverse:
            objective *= -1
        return tf.concat([z1, z2], axis=3), objective


def rational_quadratic_coupling(name,
                                x,
                                mid_channels = 512,
                                activation = "relu",
                                reverse = False,
                                dd_init=False,
                                nbins=8,
                                dropout = 0.0):
    """Reversible rational quadratic coupling layer.

    Args:
      name: variable scope.
      x: 4-D Tensor.
      mid_channels: number of channels in the coupling layer.
      activation: Can be either "relu" or "gatu".
      reverse: Forward or reverse operation.
      dropout: default, 0.0
    Returns:
      output: input s
      objective: log-determinant of the jacobian
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
        z1 = x1

        range_min = -3
        num_x2_channels = shape_list(x2)[-1]
        n_channels_out = (2 * nbins * num_x2_channels) + ((nbins-1) * num_x2_channels)

        out_params = conv_stack(
            "nn", x1, mid_channels, n_channels_out, activation=activation, dd_init=dd_init, dropout=dropout)

        bin_widths = out_params[:, :, :, :nbins*num_x2_channels]
        bin_heights = out_params[:, :, :, nbins*num_x2_channels:2*nbins*num_x2_channels]
        slopes = out_params[:, :, :, 2*nbins*num_x2_channels:]

        def _bin_act(w):
            w = tf.reshape(w, shape_list(w)[:-1] + [num_x2_channels, nbins])
            interval_length = 2*np.abs(range_min)
            return tf.math.softmax(w, axis=-1) * (interval_length - (nbins * 1e-3)) + 1e-3

        def _slope_act(w):
            w = tf.reshape(w, shape_list(w)[:-1] + [num_x2_channels, nbins-1])
            const = np.log(np.exp(1 - 1e-3) - 1)
            return tf.math.softplus(w + const) + 1e-3

        bijector = tfb.RationalQuadraticSpline(
            bin_widths=_bin_act(bin_widths),
            bin_heights=_bin_act(bin_heights),
            knot_slopes=_slope_act(slopes),
            range_min=range_min
        )

        if reverse:
            z2 = bijector.forward(x2)  # 'forward' in tfp corresponds to 'reverse' in T2T codebase
            objective = bijector.forward_log_det_jacobian(x2, event_ndims=3)
        else:
            z2 = bijector.inverse(x2)
            objective = bijector.inverse_log_det_jacobian(x2, event_ndims=3)

        return tf.concat([z1, z2], axis=3), objective


def single_conv_dist(name, x, output_channels = None):
    """A 3x3 convolution mapping x to a standard normal distribution at init.

    Args:
      name: variable scope.
      x: 4-D Tensor.
      output_channels: number of channels of the mean and std.
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x_shape = shape_list(x)
        if output_channels is None:
            output_channels = x_shape[-1]
        mean_log_scale = conv("conv2d", x, output_channels=2 * output_channels,
                              conv_init="zeros", apply_actnorm=False)
        mean = mean_log_scale[:, :, :, 0::2]
        log_scale = mean_log_scale[:, :, :, 1::2]
        return tfp.distributions.Normal(mean, tf.exp(log_scale))


def compute_prior(name, z, temperature = 1.0):
    """Distribution on z_t conditioned on z_{t-1}

    Args:
      name: variable scope.
      z: 4-D Tensor.
      temperature: float, temperature with which to sample from the Gaussian.
    Returns:
      prior_dist: instance of tfp.distributions.Normal
      state: Returns updated state.
    Raises:
      ValueError: If hparams.latent_dist_encoder is "pointwise" and if the shape
                  of latent is different from z.
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        prior_dist = single_conv_dist("level_prior", z)
        prior_mean, prior_scale = prior_dist.loc, prior_dist.scale

        mean, scale = prior_mean, prior_scale
        dist = TemperedNormal(mean, scale, temperature)

        return dist



def revnet_step(name, x, hparams, reverse=True, init=False):
    """One step of glow generative flow.

    Actnorm + invertible 1X1 conv + affine_coupling.

    Args:
      name: used for variable scope.
      x: input
      hparams: coupling_width is the only hparam that is being used in
               this function.
      reverse: forward or reverse pass.
    Returns:
      z: Output of one step of reversible flow.
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if hparams.coupling == "additive":
            coupling_layer = functools.partial(
                additive_coupling,
                name="additive",
                reverse=reverse,
                mid_channels=hparams.coupling_width,
                activation=hparams.activation,
                dd_init=init,
                dropout=hparams.coupling_dropout)
        elif hparams.coupling == "affine":
            coupling_layer = functools.partial(
                affine_coupling,
                name="affine",
                reverse=reverse,
                mid_channels=hparams.coupling_width,
                activation=hparams.activation,
                dd_init=init,
                dropout=hparams.coupling_dropout)
        elif hparams.coupling == "rational_quadratic":
            coupling_layer = functools.partial(
                rational_quadratic_coupling,
                name="rational_quadratic",
                reverse=reverse,
                mid_channels=hparams.coupling_width,
                activation=hparams.activation,
                dd_init=init,
                nbins=hparams.num_spline_bins,
                dropout=hparams.coupling_dropout)
        else:
            raise NotImplementedError

        ops = [
            functools.partial(actnorm, name="actnorm", reverse=reverse, init=init),
            functools.partial(invertible_1x1_conv, name="invertible", reverse=reverse),
            coupling_layer
        ]

        if reverse:
            ops = ops[::-1]

        objective = 0.0
        for op in ops:
            x, curr_obj = op(x=x)
            objective += curr_obj

        return x, objective


def revnet(name, x, hparams, reverse=True, init=False):
    """'hparams.depth' steps of generative flow."""
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        steps = np.arange(hparams.depth)
        objective = 0.0

        if reverse:
            steps = steps[::-1]

        if reverse:
            x, obj = invertible_1x1_conv("invertible", x, reverse=reverse)
            objective += obj

        for step in steps:
            x, obj = revnet_step("revnet_step_%d" % step, x, hparams, reverse=reverse, init=init)
            objective += obj

        if not reverse:
            x, obj = invertible_1x1_conv("invertible", x, reverse=reverse)
            objective += obj

        return x, objective


# noinspection PyIncorrectDocstring

def encoder_decoder(name, u, hparams, reverse=False, init=False, temperature=1.0):
    """Glow encoder-decoder. n_levels of (Squeeze + Flow + Split.) operations.

    Args:
        u: either x (data) or z (latent)
    """
    reshape = tfb.Reshape(event_shape_out=hparams.img_shape)

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if reverse:
            u = reshape.forward(u)
            return decode(u, hparams, init=init, temperature=temperature)
        else:
            z, objective = encode(u, hparams, init=init)
            return reshape.inverse(z), objective


def scale_x_with_alpha(x, alpha):
    return alpha + (1 - 2*alpha)*x


def apply_logit(x, objective, hparams):

    s = lambda x: scale_x_with_alpha(tf.reshape(x, [shape_list(x)[0], -1]), hparams.logit_alpha)
    get_logit_ldj = lambda x: tf.reduce_sum(tf.log(1 / s(x) + 1 / (1 - s(x))) + tf.log(1 - 2 * hparams.logit_alpha), axis=-1)
    objective += get_logit_ldj(x)

    x = scale_x_with_alpha(x, hparams.logit_alpha)
    x = tf.log(x/(1.0-x))

    return x, objective


def apply_inv_logit(x, objective, hparams):
    x = 1 / (1 + tf.exp(-x))
    x = (x - hparams.logit_alpha) / (1 - 2 * hparams.logit_alpha)

    s = lambda x: scale_x_with_alpha(tf.reshape(x, [shape_list(x)[0], -1]), hparams.logit_alpha)
    get_logit_ldj = lambda x: tf.reduce_sum(tf.log(1 / s(x) + 1 / (1 - s(x))) + tf.log(1 - 2 * hparams.logit_alpha), axis=-1)
    objective -= get_logit_ldj(x)

    return x, objective


def encode(x, hparams, init=False):
    objective = 0.0
    all_latents = []

    if hparams.logit_alpha:
        x += hparams.shift
        x, objective = apply_logit(x, objective, hparams)
        x -= hparams.logit_shift

    # Squeeze + Flow + Split
    for level in range(hparams.n_levels):
        x = squeeze(x, hparams.img_shape, level=level, reverse=False)

        x, obj = revnet("revnet_%d" % level, x, hparams, reverse=False, init=init)
        objective += obj

        if level < hparams.n_levels - 1 and hparams.use_split:
            x, obj, z = split("split_%d" % level, x, reverse=False)
            objective += obj
            all_latents.append(z)

    if hparams.use_split:
        final_z = x
        all_latents.append(final_z)
        z = stack_latents(all_latents, hparams.img_shape)  # combine all latents into one tensor
    else:
        z = x
        for level in reversed(range(hparams.n_levels)):
            z = squeeze(z, hparams.img_shape, level=level, reverse=True)

    return z, objective


def decode(z, hparams, init, temperature):

    all_latents, z = get_init_decode_z(hparams, z)
    objective = 0.0

    for level in reversed(range(hparams.n_levels)):

        if level < hparams.n_levels - 1 and hparams.use_split:
            z, obj = split("split_%d" % level, z, z=all_latents[level], reverse=True, temperature=temperature)
            objective += obj

        z, obj = revnet("revnet_%d" % level, z, hparams=hparams, reverse=True, init=init)
        objective += obj
        z = squeeze(z, hparams.img_shape, level=level, reverse=True)

    x = z
    if hparams.logit_alpha:
        x += hparams.logit_shift
        x, objective = apply_inv_logit(x, objective, hparams)
        x -= hparams.shift

    return x, objective


def get_init_decode_z(hparams, z):

    if hparams.use_split:
        all_latents = unstack_latents(z, hparams.n_levels, hparams.img_shape)
        z = all_latents[-1]
    else:
        all_latents = None
        for level in range(hparams.n_levels):
            z = squeeze(z, hparams.img_shape, level=level, reverse=False)

    return all_latents, z


def stack_latents(z_list, img_shape):
    """Stack list of (differently shaped) latents produced by multi-scale glow into one tensor

    Given an input image x, Glow uses multiple squeeze() and split() operations, yielding a sequence of
    latent z vars of varying shapes. This method takes a list of those z vars (where the last
    element in the list corresponds to the final z produced after all the squeezes and splits) and
    re-assembles them to produce a single tensor with the same shape as the input image"""
    n_z = len(z_list)
    z1 = z_list[-1]
    for i in reversed(range(n_z)):
        z2 = z_list[i]
        if i < n_z - 1:
            z1 = tf.concat([z1, z2], 3)  # unsplit
        z1 = squeeze(z1, img_shape, level=i, reverse=True)  # unsqueeze
    return z1


def unstack_latents(z1, n_levels, img_shape):
    """Inverse method of stack_latents() - see docstring there"""
    unstacked_latents = []
    for i in range(n_levels):
        z1 = squeeze(z1, img_shape, level=i, reverse=False)  # squeeze
        if i < n_levels - 1:
            z1, z2 = tf.split(z1, num_or_size_splits=2, axis=-1)  # split
            unstacked_latents.append(z2)
    unstacked_latents.append(z1)
    return unstacked_latents


def split(name, x, reverse=False, z=None, temperature=1.0):
    """Splits / concatenates x into x1 and x2 across number of channels.

  For the forward pass, x2 is assumed be gaussian,
  i.e P(x2 | x1) ~ N(mu, sigma) where mu and sigma are the outputs of
  a network conditioned on x1 and optionally on cond_latents.
  For the reverse pass, x2 is determined from mu(x1) and sigma(x1).
  This is deterministic/stochastic depending on whether eps is provided.

  Args:
    name: variable scope.
    x: 4-D Tensor, shape (NHWC).
    reverse: Forward or reverse pass.
    z: If eps is provided, x2 is set to be
    eps_std: Sample x2 with the provided eps_std.
    hparams: next_frame_glow hparams.
    temperature: Temperature with which to sample from the gaussian.

  Returns:
  Raises:
    ValueError: If latent is provided and shape is not equal to NHW(C/2)
                where (NHWC) is the size of x.
  """

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        if reverse:
            prior_dist = compute_prior("prior_on_z2", x, temperature=temperature)
            x2 = set_eps(prior_dist, z)
            ldj = tf.reduce_sum(tf.log(prior_dist.scale), axis=[1, 2, 3])
            return tf.concat([x, x2], 3), ldj

        else:
            x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)

            # objective: P(x2|x1) ~ N(x2 ; NN(x1))
            prior_dist = compute_prior("prior_on_z2", x1)
            ildj = -tf.reduce_sum(tf.log(prior_dist.scale), axis=[1, 2, 3])
            z = get_eps(prior_dist, x2)
            return x1, ildj, z


def squeeze(x, img_shape, level=0, reverse = True):
    """Block-wise spatial squeezing of x to increase the number of channels.

    Args:
      name: Used for variable scoping.
      x: 4-D Tensor of shape (batch_size X H X W X C)
      reverse: Squeeze or unsqueeze operation.

    Returns:
      x: 4-D Tensor of shape (batch_size X (H//factor) X (W//factor) X
         (cXfactor^2). If reverse is True, then it is factor = (1 / factor)
    """
    factor = 2

    h, w, c = img_shape
    assert h % (factor**level) == 0 and w % (factor**level) == 0
    height = h // (factor**level)
    width = w // (factor**level)
    n_channels = c * (factor**(2*level))

    if not reverse:
        x = tf.reshape(x, [-1, height // factor, factor, width // factor, factor, n_channels])
        x = tf.transpose(x, [0, 1, 3, 5, 2, 4])
        x = tf.reshape(x, [-1, height // factor, width // factor, int(n_channels * (factor**2))])
    else:
        x = tf.reshape(x, (-1, height // 2, width // 2, n_channels, factor, factor))
        x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
        x = tf.reshape(x, (-1, height, width, n_channels))
    return x
