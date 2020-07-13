import tensorflow_probability as tfp

from utils.misc_utils import *
from utils.tf_utils import *

tfd = tfp.distributions


class MogMade:
    """This code is adapted from native tensorflow code.

    See:
    https://github.com/tensorflow/probability/blob/master/tensorflow_probability/
    python/bijectors/masked_autoregressive.py
    """

    def __init__(self, base_dist, mog_param_fn, n_dims, name=None):
        """Creates the MaskedAutoregressiveFlow bijector.

        Args:
            mog_param_fn: Python `callable` which computes `shift` and
              `log_scale` and `log unnormalised mixture probability`
               from both the forward domain (`x`) and the inverse domain
              (`y`). Calculation must respect the "autoregressive property" (see class
              docstring). Suggested default
              `masked_autoregressive_default_template(hidden_layers=...)`.
              Typically the function contains `tf.Variables` and is wrapped using
              `tf.make_template`. Returning `None` for either (both) `shift`,
              `log_scale` is equivalent to (but more efficient than) returning zero.
            name: Python `str`, name given to ops managed by this object.
        """
        self.name = name or "mog_made"
        self.base_dist = base_dist
        self._mog_param_fn = mog_param_fn
        self.n_dims = n_dims


    def _inverse(self, x, return_params=False):
        raise NotImplementedError

    def _inverse_log_det_jacobian(self, x):
        raise NotImplementedError

    def log_prob(self, x):
        x = tf.reshape(x, [-1, self.n_dims])
        logits, means, scales = self._mog_param_fn(x)  # each of shape (N, D, C)

        log_prob = tf.reduce_sum(
            tf.reduce_logsumexp(
                logits - 0.5 *
                (
                        np.log(2 * np.pi)
                        + 2 * tf.log(scales)
                        + ((tf.reshape(x, shape_list(x) + [-1]) - means) / scales) ** 2
                ),
                axis=-1
            ),
            axis=-1
        )

        return log_prob

    def sample(self, shape, seed):

        N, D = shape, self.n_dims
        samples = tf.zeros((N, D))  # (N, D)
        # samples = tf.random_normal((N, D))  # (N, D)

        for i in range(self.n_dims):
            logits, means, scales = self._mog_param_fn(samples)  # each of shape (N, D, C)

            logits, means, scales = (
                logits[:, i, :],
                means[:, i, :],
                scales[:, i, :]  # each of shape (N, C)
            )

            mixture_distribution = tfd.Categorical(logits=logits)
            components = tf.reshape(mixture_distribution.sample((1,)), [-1, 1])  # (N, 1)

            means, scales = (
                tf.gather(means, components, axis=1, batch_dims=1),
                tf.gather(scales, components, axis=1, batch_dims=1)  # each of shape (N, 1)
            )

            update = means + tf.random_normal((N, 1)) * scales
            samples = tf.concat([samples[:, :i], update, samples[:, i+1:]], axis=1)

        return samples


def get_mask(in_features, out_features, autoregressive_features, mask_type=None):
    max_ = max(1, autoregressive_features - 1)
    min_ = min(1, autoregressive_features - 1)

    if mask_type == 'input':
        in_degrees = np.arange(1, autoregressive_features + 1)
        out_degrees = np.arange(out_features) % max_ + min_
        mask = (out_degrees[..., None] >= in_degrees)

    elif mask_type == 'output':
        in_degrees = np.arange(in_features) % max_ + min_
        out_degrees = np.repeat(
            np.arange(1, autoregressive_features + 1),
            out_features // autoregressive_features
        )
        mask = (out_degrees[..., None] > in_degrees)

    else:
        in_degrees = np.arange(in_features) % max_ + min_
        out_degrees = np.arange(out_features) % max_ + min_
        mask = (out_degrees[..., None] >= in_degrees)

    return mask.astype('float32')


def masked_dense(inputs,
                 units,
                 num_blocks=None,
                 mask_type=None,
                 kernel_initializer=None,
                 reuse=None,
                 name=None,
                 activation=None,
                 *args,
                 **kwargs):

    input_depth = inputs.shape.with_rank_at_least(1)[-1].value
    if input_depth is None:
        raise NotImplementedError("Rightmost dimension must be known prior to graph execution.")

    mask = get_mask(input_depth, units, num_blocks, mask_type).T

    if kernel_initializer is None:
        kernel_initializer = tf.glorot_normal_initializer()

    def masked_initializer(shape, dtype=None, partition_info=None):
        return mask * kernel_initializer(shape, dtype, partition_info)

    with tf.name_scope(name, "masked_dense", [inputs, units, num_blocks]):
        layer = tf.layers.Dense(
            units,
            kernel_initializer=masked_initializer,
            kernel_constraint=lambda x: mask * x,
            name=name,
            dtype=inputs.dtype.base_dtype,
            _scope=name,
            _reuse=reuse,
            *args,
            **kwargs
            )

    return layer.apply(inputs)


def masked_residual_block(inputs, num_blocks, activation=tf.nn.relu, dropout_keep_p=1., *args, **kwargs):
    input_depth = inputs.get_shape().as_list()[-1]

    # First residual layer
    residual = inputs
    residual = activation(residual)
    residual = masked_dense(
        residual, input_depth, num_blocks,
        kernel_initializer=tf.variance_scaling_initializer(scale=2., distribution='normal'),
        *args, **kwargs
    )

    # Second residual layer
    residual = activation(residual)
    residual = tf.nn.dropout(residual, keep_prob=dropout_keep_p)
    residual = masked_dense(
        residual, input_depth, num_blocks,
        kernel_initializer=tf.variance_scaling_initializer(scale=0.1, distribution='normal'),
        *args, **kwargs
    )
    return inputs + residual


def residual_made_net(x, n_out=2, n_residual_blocks=2, hidden_units=256, activation=tf.nn.relu, dropout_keep_p=1.):

    input_depth = x.get_shape().as_list()[-1]

    output = masked_dense(inputs=x, units=hidden_units, num_blocks=input_depth, mask_type='input')

    for _ in range(n_residual_blocks):
        output = masked_residual_block(
            inputs=output, num_blocks=input_depth, activation=activation, dropout_keep_p=dropout_keep_p
        )

    output = activation(output)
    output = masked_dense(
        inputs=output,
        units=n_out*input_depth,
        num_blocks=input_depth,
        activation=None,
        mask_type='output',
        bias_initializer=tf.glorot_normal_initializer()
    )

    return tf.reshape(output, [-1, input_depth, n_out])


def residual_made_template(n_out=2,
                           n_residual_blocks=2,
                           hidden_units=256,
                           activation=tf.nn.relu,
                           dropout_keep_p=1.,
                           log_scale_clip_gradient=False,
                           scale_min_clip=-5.,
                           scale_max_clip=20.):
    def _fn(x):
        output = residual_made_net(x,
                                   n_out=n_out,
                                   n_residual_blocks=n_residual_blocks,
                                   hidden_units=hidden_units,
                                   activation=activation,
                                   dropout_keep_p=dropout_keep_p)
        shift, presoft_scale = tf.unstack(output, num=2, axis=-1)

        which_clip = (tf.clip_by_value if log_scale_clip_gradient else _clip_by_value_preserve_grad)
        presoft_scale = which_clip(presoft_scale, scale_min_clip, scale_max_clip)

        scale = 1e-3 + tf.nn.softplus(presoft_scale + np.log(np.e - 1))

        return shift, tf.log(scale)\

    return tf.make_template("residual_made_template", _fn)


def residual_mog_made_template(n_out=30,
                               n_residual_blocks=2,
                               hidden_units=256,
                               activation=tf.nn.relu,
                               dropout_keep_p=1.,
                               scale_min=1e-3
                               ):
    def _fn(x):
        output = residual_made_net(x,
                                   n_out=n_out,
                                   n_residual_blocks=n_residual_blocks,
                                   hidden_units=hidden_units,
                                   activation=activation,
                                   dropout_keep_p=dropout_keep_p)
        n_mixture_comps = n_out // 3

        logits = output[..., : n_mixture_comps]
        means = output[..., n_mixture_comps: 2 * n_mixture_comps]
        scales = output[..., 2 * n_mixture_comps:]

        # create extra parameter to ensure that logits are inited near 0
        alpha = tf.get_variable("alpha", shape_list(logits)[1:], initializer=tf.random_normal_initializer(1e-2))

        logits = tf.nn.log_softmax(alpha * logits, dim=-1)
        scales = tf.nn.softplus(np.log(np.e - 1) + scales) + scale_min  # initialised to near 1

        return logits, means, scales  # (n_samples, n_features, n_components)

    return tf.make_template("residual_made_template", _fn)


def _clip_by_value_preserve_grad(x, clip_value_min, clip_value_max, name=None):
    """Clips input while leaving gradient unaltered."""
    with tf.name_scope(name, "clip_by_value_preserve_grad", [x, clip_value_min, clip_value_max]):
        clip_x = tf.clip_by_value(x, clip_value_min, clip_value_max)
        return x + tf.stop_gradient(clip_x - x)
