import functools
import logging
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from keras_layers import GatuOrTanh
from tensorflow.keras import layers as k_layers
from tensorflow.keras import initializers
from __init__ import project_root


def tf_batched_operation(sess, ops, n_samples, batch_size, data_pholder=None, data=None, const_feed_dict=None, feed_dict_fn=None):
    if not isinstance(ops, list):
        ops = [ops]
    output = [[] for _ in range(len(ops))]

    if feed_dict_fn is None and data_pholder is None and const_feed_dict is None:
        raise ValueError("must specify at least one of 'feed_dict_fn', 'data_pholder' or 'const_feed_dict'")

    if feed_dict_fn is None and data_pholder is not None:
        def feed_dict_fn(j, n, b=batch_size):
            fd = const_feed_dict if const_feed_dict else {}
            fd.update({data_pholder: data[j:min(j+b, n), ...]})
            return fd

    elif feed_dict_fn is None and const_feed_dict is not None:
        def feed_dict_fn(j, n, b=batch_size):
            return const_feed_dict

    for k in range(0, n_samples, batch_size):
        fetch = sess.run(ops, feed_dict=feed_dict_fn(k, n_samples, batch_size))
        for i in range(len(ops)):
            output[i].append(fetch[i])

    concat_outputs = []
    for i in range(len(ops)):
        if isinstance(output[i][0], np.ndarray):
            concat_outputs.append(np.concatenate(output[i]))
        else:
            concat_outputs.append(np.array(output[i]))

    if len(ops) == 1:
        return concat_outputs[0]
    else:
        return concat_outputs


def tf_product(a, b):
    """Tensorflow equivalent of itertools.product in python"""
    a, b = a[:, None, None], b[None, :, None]
    return tf.concat([a + tf.zeros_like(b), tf.zeros_like(a) + b], axis=2)


def tf_log_mean_exp(x):
    n = tf.cast(shape_list(x)[0], dtype=tf.float32)
    return tf.reduce_logsumexp(x) - tf.log(n)


def tf_log_var_exp(x):
    """Given x=log(w), compute log(var(w)) using numerically stable operations"""
    mu = tf_log_mean_exp(x)
    x_max = tf.reduce_max(x)
    x_prime = x - x_max
    mu_prime = mu - x_max
    summand = tf.exp(2 * x_prime) - tf.exp(2 * mu_prime)
    n = tf.cast(shape_list(x)[0], dtype=tf.float32)
    return 2 * x_max + tf.log(tf.reduce_sum(summand)) - tf.log(n)


def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def get_tf_activation(act_name):
    """Returns keras activation function from its name"""

    if act_name == "leaky_relu":
        activation = k_layers.LeakyReLU
    elif act_name == "gatu":
        activation = GatuOrTanh  # gatu for 4D inputs, otherwise Relu
    else:
        activation = lambda: k_layers.Activation(act_name)

    return activation


def make_block_diagonal(matrices, dtype=tf.float32):
    """Constructs block-diagonal matrices from a list of batched 2D tensors.

    Taken from: https://stackoverflow.com/questions/42157781/block-diagonal-matrices-in-tensorflow

    Args:
        matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
          matrices with the same batch dimension).
    dtype: Data type to use. The Tensors in `matrices` must match this dtype.

    Returns:
        A matrix with the input matrices stacked along its main diagonal, having
        shape [..., \sum_i N_i, \sum_i M_i].
    """
    matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
    blocked_rows = tf.Dimension(0)
    blocked_cols = tf.Dimension(0)
    batch_shape = tf.TensorShape(None)
    for matrix in matrices:
        full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
        batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
        blocked_rows += full_matrix_shape[-2]
        blocked_cols += full_matrix_shape[-1]
    ret_columns_list = []
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        ret_columns_list.append(matrix_shape[-1])
    ret_columns = tf.add_n(ret_columns_list)
    row_blocks = []
    current_column = 0
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        row_before_length = current_column
        current_column += matrix_shape[-1]
        row_after_length = ret_columns - current_column
        row_blocks.append(tf.pad(
            tensor=matrix,
            paddings=tf.concat(
                [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
                 [(row_before_length, row_after_length)]],
                axis=0)))
    blocked = tf.concat(row_blocks, -2)
    blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
    return blocked


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if no arguments are provided. All arguments must be optional.
    @misc{hafner2016scopedecorator,
      author = {Hafner, Danijar},
      title = {Structuring Your TensorFlow Models},
      year = {2016},
      howpublished = {Blog post},
      url = {https://danijar.com/structuring-your-tensorflow-models/}
    }
    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def tf_var_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    @misc{hafner2016scopedecorator,
      author = {Hafner, Danijar},
      title = {Structuring Your TensorFlow Models},
      year = {2016},
      howpublished = {Blog post},
      url = {https://danijar.com/structuring-your-tensorflow-models/}
    }
    """
    name = scope or function.__name__

    @functools.wraps(function)
    def decorator(self, *args2, **kwargs2):
        with tf.variable_scope(name, *args, **kwargs):
            return function(self, *args2, **kwargs2)

    return decorator


@doublewrap
def tf_name_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations.
    The operations added by the function live within a tf.name_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    name scope. The scope name defaults to the name of the wrapped
    function.
    @misc{hafner2016scopedecorator,
      author = {Hafner, Danijar},
      title = {Structuring Your TensorFlow Models},
      year = {2016},
      howpublished = {Blog post},
      url = {https://danijar.com/structuring-your-tensorflow-models/}
    }
    """
    name = scope or function.__name__

    @functools.wraps(function)
    def decorator(self, *args2, **kwargs2):
        with tf.name_scope(name, *args, **kwargs):
            return function(self, *args2, **kwargs2)

    return decorator


@doublewrap
def tf_cache_template(function, scope=None, *args, **kwargs):
    """
    A decorator for class methods that define TensorFlow operations.

    The function is first wrapped in a  variable_scope() that allows
    us to specify a initializer, regularizer etc.
    The result is then wrapped in a make_template() so that
    variables created within the function are reused on
    subsequent calls. The template has the same name as the
    function by default.

    The decorator accepts arguments that are passed to variable_scope().
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @functools.wraps(function)
    def decorator(self, *args1, **kwargs1):
        if not hasattr(self, attribute):
            def scoped_function(self, *args1, **kwargs1):
                with tf.variable_scope(name, *args, **kwargs):
                    return function(self, *args1, **kwargs1)

            template = tf.compat.v1.make_template(name, scoped_function)
            setattr(self, attribute, template)
        return getattr(self, attribute)(self, *args1, **kwargs1)

    return decorator


def tf_repeat_first_axis(x, n_repeat):
    """This has same functionality as np.repeat for the specific case of repeating along axis 0"""
    event_dim_shp = shape_list(x)[1:]
    x = tf.expand_dims(x, axis=1)
    tile_shape = [1, n_repeat] + [1 for _ in range(len(event_dim_shp))]
    x = tf.tile(x, tile_shape)
    x = tf.reshape(x, [-1, *event_dim_shp])
    return x


def tf_combine_shapes(shape1, shape2, axis=0):
    shape1 = maybe_make_constant_tensor(shape1)
    shape2 = maybe_make_constant_tensor(shape2)

    return tf.concat([shape1, shape2], axis=axis)


def maybe_make_constant_tensor(input):
    if isinstance(input, list):
        input = tf.constant(input)
    return input


def tf_dense(h, size, reg_coef, name):
    h_shape = shape_list(h)
    W = tf.compat.v1.get_variable("W_" + name, shape=(h_shape[-1], size), regularizer=tf_l2_regulariser(scale=reg_coef))
    b = tf.compat.v1.get_variable("b_" + name, shape=(size,), initializer=initializers.Zeros())

    if len(h_shape) > 2:
        h = tf.reshape(h, [-1, h_shape[-1]]) @ W + b
        h = tf.reshape(h, [*h_shape[:-1], size])
    else:
        h = h @ W + b

    return h


def tf_l2_regulariser(scale, scope=None):

    def l2(weights):
        """Applies l2 regularization to weights."""
        with tf.name_scope(scope, 'l2_regularizer', [weights]) as name:
            my_scale = tf.convert_to_tensor(scale,
                                            dtype=weights.dtype.base_dtype,
                                            name='scale')
            return tf.multiply(my_scale, tf.reduce_sum(weights**2), name=name)
    return l2


def tf_l2_regulariser_per_head(scales, weights):

    my_scales = tf.convert_to_tensor(scales, dtype=weights.dtype.base_dtype, name='scale')
    axes_to_sum = list(range(1, len(shape_list(weights))))

    loss = tf.tensordot(my_scales, tf.reduce_sum(weights**2, axis=axes_to_sum), 1)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, loss)


def tf_duplicate_waymarks(x, concat=True):
    x_shp = shape_list(x)
    x1 = tf.gather(x, tf.range(x_shp[1] - 1), axis=1)  # (n, num_ratios, ...)
    x2 = tf.gather(x, tf.range(1, x_shp[1]), axis=1)  # (n, num_ratios, ...)
    if concat:
        return tf.concat([x1, x2], axis=0)  # (2n, num_ratios, ...)
    else:
        return tf.stack([x1, x2], axis=0)  # (2, n, num_ratios, ...)


def tf_first_and_last_waymarks(x, concat=True):
    x_shp = shape_list(x)
    n, k, event_dims = x_shp[0], x_shp[1], x_shp[2:]
    event_ones = [1 for _ in event_dims]

    x0 = tf.gather(x, 0, axis=1)  # (n, ...)
    xk = tf.gather(x, k-1, axis=1)  # (n, ...)

    if concat:
        x0_xk = tf.concat([x0, xk], axis=0)  # (2n, ...)
        x0_xk = tf.tile(tf.reshape(x0_xk, [2*n, 1, *event_dims]), [1, k-1, *event_ones])  # (2n, n_ratios, ...)
    else:
        x0_xk = tf.stack([x0, xk], axis=0)  # (2, n, ...)
        x0_xk = tf.tile(tf.reshape(x0_xk, [2, n, 1, *event_dims]), [1, 1, k-1, *event_ones])  # (2, n, n_ratios, ...)

    return x0_xk


def crop_and_resize(x, w, h, centre=140, resize=32):
    """Centre crop to centre*centre and then resize to resize*resize"""
    w_crop_indent, h_crop_indent = int((w-centre)/2), int((h-centre)/2)
    x = x[:, w_crop_indent:w-w_crop_indent, h_crop_indent:h-h_crop_indent, :]
    x = tf.image.resize(x, tf.constant([resize, resize], dtype=tf.int32))
    return x


def sigmoid_with_a_temperature(x, temp):
    """poor sigmoid"""
    return (1 / (1 + tf.exp(-temp*x))) - 0.5


def scaled_bump_function(x, threshold):
    x_shift, x_scale = 5, 0.1
    f = lambda u: tf.exp(-1 / (1 - (x_scale*u)**2))
    z = x - x_shift
    bump = (f(z)-f(x_shift)) / f(x_shift) * threshold
    return tf.where((x_scale*z) ** 2 < 1, bump, tf.ones_like(bump) * -threshold)


def tf_keras_matmul(a, b, transpose_b=False):
    return k_layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=transpose_b))([a, b])


def tf_keras_hw_flatten(x):
    """Flattens the height & width dims of a set of feature maps"""
    return k_layers.Reshape(target_shape=[-1, x.shape[-1]])(x)


def tf_keras_transpose(x):
    return k_layers.Lambda(lambda y: K.transpose(y))(x)


def l2_norm(input_x, epsilon=1e-12, axis=None):
    """normalize input to unit norm"""
    input_x_norm = input_x/(tf.reduce_sum(input_x**2, axis=axis, keepdims=True)**0.5 + epsilon)
    return input_x_norm


def make_flexible_Conv2D_fn():

    def flex_conv2d(*args, just_track_spectral_norm=False, **kwargs):
        return k_layers.Conv2D(*args, **kwargs)

    return flex_conv2d


def make_flexible_Dense_fn():

    def flex_dense(*args, just_track_spectral_norm=False, **kwargs):
        return k_layers.Dense(*args, **kwargs)

    return flex_dense


def tf_enforce_lower_diag_and_nonneg_diag(A, shift=0.0):
    mask = tf.ones_like(A)
    ldiag_mask = tf.matrix_band_part(mask, -1, 0)
    diag_mask = tf.matrix_band_part(mask, 0, 0)
    strict_ldiag_mask = ldiag_mask - diag_mask

    # should I use exp or softplus here?
    return strict_ldiag_mask * A + diag_mask * tf.exp(A - shift)


def tf_enforce_symmetric_and_pos_diag(A, shift=0.0):

    mask = tf.ones_like(A)
    ldiag_mask = tf.matrix_band_part(mask, -1, 0)
    diag_mask = tf.matrix_band_part(mask, 0, 0)
    strict_ldiag_mask = ldiag_mask - diag_mask

    B = strict_ldiag_mask * A
    if len(A.get_shape().as_list()) == 3:
        B += tf.transpose(B, [0, 2, 1])
    else:
        B += tf.transpose(B)

    B += diag_mask * tf.exp(A - shift)

    return B


def tf_print_total_number_of_trainable_params():
    """prints total number of trainable parameters according to default graph"""
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print(total_parameters)


def tf_spectral_norm(w, u, iteration=1, training=True, max_norm=1):
    w_shape = shape_list(w)
    w = tf.reshape(w, [-1, w_shape[-1]])

    u_hat = tf.identity(u)
    for i in range(iteration):
        # power iteration: usually iteration = 1 will be enough

        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    sigma = tf.maximum(sigma / max_norm, 1)

    if training:
        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = w / sigma
    else:
        w_norm = w / sigma

    w_norm = tf.reshape(w_norm, w_shape)
    return w_norm


def tf_batched_spectral_norm(w, u, iteration=1, training=True, max_norms=1, extract_idxs=None):

    if not tf.is_tensor(max_norms): max_norms = tf.convert_to_tensor(max_norms)
    max_norms = tf.reshape(max_norms, [-1, 1, 1])  # (k, 1, 1)

    if extract_idxs is not None:
        u_hat = tf.gather(u, extract_idxs, axis=0)  # (k, 1, input_dim)
    else:
        u_hat = tf.identity(u)  # (k, 1, d2)

    w_tr = tf.transpose(w, [0, 2, 1])  # (k, d2, d1)

    for i in range(iteration):
        # power iteration: usually iteration = 1 will be enough

        v_ = tf.matmul(u_hat, w_tr)  # (k, 1, d1)
        v_hat = l2_norm(v_, axis=[1, 2])  # (k, 1, d1)

        u_ = tf.matmul(v_hat, w)  # (k, 1, d2)
        u_hat = l2_norm(u_, axis=[1, 2])  # (k, 1, d2)

    A = tf.matmul(v_hat, w)  # (k, 1, d2)
    sigma = tf.matmul(A, tf.transpose(u_hat, [0, 2, 1]))  # (k, 1, 1)
    sigma = tf.maximum(sigma / max_norms, 1)

    if extract_idxs is not None:
        u_hat_big = tf.scatter_nd(indices=tf.reshape(extract_idxs, [-1, 1]), updates=u_hat, shape=shape_list(u))
        u_hat = u_hat_big

    if training:
        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = w / sigma
    else:
        w_norm = w / sigma

    return w_norm  # (k, d1, d2)

def tf_get_power_seq(start, stop, p, n):
    print(np.linspace(start ** (1 / p), stop ** (1 / p), n) ** p)
    return tf.linspace(start ** (1 / p), stop ** (1 / p), n) ** p


def tf_spatially_arrange_imgs(x, n_imgs, wmark_input=True):
    """assuming inputs x are stacked along channels, rearrange them into a spatial grid"""
    n_sqrt = int(n_imgs**0.5)
    if wmark_input:
        transpose_idxs = [0, 1, 4, 2, 3]
        split_ax = 2
        batch_dims = shape_list(x)[:2]
    else:
        transpose_idxs = [0, 3, 1, 2]
        split_ax = 1
        batch_dims = shape_list(x)[:1]

    x = tf.transpose(x, transpose_idxs)  # (*batch_dims, n_sqrt**2, 28, 28)
    sub_imgs = tf.split(x, n_sqrt, axis=split_ax)  # n_sqrt tensors of shape (*batch_dims, n_sqrt, 28, 28)
    sub_imgs = [tf.reshape(i, [*batch_dims, 28 * n_sqrt, 28]) for i in sub_imgs]
    x = tf.concat(sub_imgs, axis=-1)  # (*batch_dims, 28*n_sqrt, 28*n_sqrt)
    return tf.expand_dims(x, -1)


def tf_unarrange_spatial_imgs(x, n_imgs, wmark_input=True):
    """assuming input(s) are images spatially stacked together, rearrange them to be stacked channel-wise"""
    n_sqrt = int(n_imgs ** 0.5)
    # input has shape: (*batch_dims, n_sqrt * 28, n_sqrt * 28, 1)
    if wmark_input:
        transpose_idxs = [0, 1, 3, 4, 2]
        concat_ax = 2
        batch_dims = shape_list(x)[:2]
    else:
        transpose_idxs = [0, 2, 3, 1]
        concat_ax = 1
        batch_dims = shape_list(x)[:1]

    sub_imgs = tf.split(x, n_sqrt, axis=-2)  # n_sqrt tensors of shape (*batch_dims, n_sqrt*28, 28, 1)
    sub_imgs = [tf.reshape(i, [*batch_dims, n_sqrt, 28, 28]) for i in sub_imgs]  # n_sqrt tensors: (*batch_dims, n_sqrt, 28, 28)
    x = tf.concat(sub_imgs, axis=concat_ax)  # (*batch_dims, n_sqrt**2, 28, 28)
    x = tf.transpose(x, transpose_idxs)  # (*batch_dims, 28, 28, n_sqrt**2)
    return x


def tf_lbfgs_function_factory(model, loss, train_x, trace_fn=None, trace_label=None):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.

    Code taken from:
     https://pychao.com/2019/11/02/optimize-tensorflow-keras-models-with-l-bfgs-from-tensorflow-probability/

    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
        train_x [in]: the input part of training data.

    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    shapes = [shape_list(p) for p in model.trainable_variables]
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = np.prod(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)

    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.

        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """
        assign_ops = []
        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            assign_ops.append(model.trainable_variables[i].assign(tf.reshape(param, shape)))
        return assign_ops

    # now create a function that will be returned by this factory
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.

        This function is created by function_factory.

        Args:
           params_1d [in]: a 1D tf.Tensor.

        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:

            # update the parameters in the model
            with tf.control_dependencies(assign_new_model_parameters(params_1d)):
                # calculate the loss
                model_output = model(train_x, training=True)
                loss_value = loss(model_output)

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        f.iter.assign_add(1)
        tf.print("Iter:", f.iter, "loss:", loss_value)

        if trace_fn is not None:
            trace_val = trace_fn(model_output)
            tf.print("{}:".format(trace_label), trace_val)

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters

    return f
