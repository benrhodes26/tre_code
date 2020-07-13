import tensorflow_probability as tfp

from density_estimators.flows import Flow
from tensorflow import layers
from tensorflow.keras.models import Model
from tensorflow.keras import layers as k_layers
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from utils.keras_layers import *
from utils.misc_utils import *
from utils.tf_utils import *

tfd = tfp.distributions


class LinearHeads:
    """Final linear heads in a multi-classifier energy network"""

    def __init__(self,
                 input_dim,
                 bridge_idxs,
                 max_num_ratios,
                 use_bias=True,
                 use_single_head=False,
                 head_multiplier=1.0,
                 max_spectral_norm_params=None
                 ):

        if max_spectral_norm_params is not None:
            raise NotImplementedError("Haven't implemented spectral normalisation for linear heads")

        self.input_dim = input_dim
        self.bridge_idxs = bridge_idxs
        self.max_num_ratios = max_num_ratios
        self.use_bias = use_bias
        self.use_single_head = use_single_head
        self.head_multiplier = head_multiplier
        self.head_assignments = tf.no_op()

    @tf_cache_template("linear_heads", initializer=initializers.glorot_normal())
    def eval(self, x, is_train, is_wmark_input=False):

        W, b = self.get_weights(is_train)

        e = tf.reduce_sum(x * W, axis=-1)  # (?, num_ratios)

        if self.use_bias:
            e += b  # (?, num_ratios)

        return -e  # (?, num_ratios) - neg energies

    def get_weights(self, is_train):

        W_shape = self.input_dim if self.use_single_head else (self.max_num_ratios, self.input_dim)
        W_all = tf.compat.v1.get_variable("W_all", shape=W_shape)

        if not self.use_single_head:
            W = tf.gather(W_all, self.bridge_idxs, axis=0)  # (num_ratios, input_dim)

        if self.use_bias:
            b_shape = () if self.use_single_head else self.max_num_ratios
            b_all = tf.compat.v1.get_variable("b_all", shape=b_shape, initializer=initializers.Zeros())
            if not self.use_single_head:
                b = tf.gather(b_all, self.bridge_idxs, axis=0)  # (num_ratios,)
        else:
            b = None

        return W, b

    def neg_energy(self, x, is_train, is_wmark_input=False):
        x = ready_x_for_per_bridge_computation(x, is_wmark_input, self.bridge_idxs)
        neg_e = self.eval(x, is_train, is_wmark_input=is_wmark_input)
        return neg_e


class QuadraticHeads:
    """Final quadratic layer in a multi-classifier energy network"""

    def __init__(self,
                 input_dim,
                 bridge_idxs,
                 max_num_ratios,
                 use_single_head=False,
                 max_spectral_norm_params=None,
                 quadratic_constraint_type="semi_pos_def",
                 use_linear_term=True,
                 reg_coef=0.
                 ):

        self.input_dim = input_dim
        self._bridge_idxs = bridge_idxs
        self.max_num_ratios = max_num_ratios
        self.use_single_head = use_single_head
        self.max_spectral_norm_params = max_spectral_norm_params
        self.quadratic_constraint_type = quadratic_constraint_type
        self.use_linear_term = use_linear_term
        self.reg_coef = reg_coef

        self.head_assignments = tf.no_op()

    @property
    def bridge_idxs(self):
        return self._bridge_idxs

    @bridge_idxs.setter
    def bridge_idxs(self, val):
        self._bridge_idxs = val

    @tf_cache_template("quad_heads", initializer=initializers.normal(stddev=0.001))
    def eval(self, x, is_train, is_wmark_input=False, *args):

        # x has shape (?, num_ratios, d)
        Q, W, b = self.get_weights(is_train)

        e = 0
        e += self.quad_energy(x, Q)  # (?, num_ratios)
        if self.use_linear_term:
                e += tf.reduce_sum(x * W, axis=-1)  # (?, num_ratios)
        e += b

        return -e  # (?, num_ratios)

    def get_weights(self, is_train):
        if self.use_single_head:
            Q_shape = (self.input_dim, self.input_dim)
            W_shape = self.input_dim
        else:
            Q_shape = (self.max_num_ratios, self.input_dim, self.input_dim)
            W_shape = (self.max_num_ratios, self.input_dim)

        unconstrained_Q = tf.compat.v1.get_variable("Q_all",
                                                    shape=Q_shape,
                                                    initializer=initializers.Zeros(),
                                                    regularizer=tf_l2_regulariser(scale=self.reg_coef),
                                                    dtype=tf.float32)
        W = tf.compat.v1.get_variable("W_all", initializer=initializers.Zeros(), shape=W_shape)
        b = tf.compat.v1.get_variable("b_all", shape=self.max_num_ratios, initializer=initializers.Zeros())

        if not self.use_single_head:
            unconstrained_Q = tf.gather(unconstrained_Q, self.bridge_idxs, axis=0)  # (num_ratios, input_dim, input_dim)
            W = tf.gather(W, self.bridge_idxs, axis=0)  # (num_ratios, input_dim)
        b = tf.gather(b, self.bridge_idxs, axis=0)  # (num_ratios, )

        if self.quadratic_constraint_type == "semi_pos_def":
            L = tf_enforce_lower_diag_and_nonneg_diag(unconstrained_Q, shift=5.0)
            Q = tf.matmul(L, tf.transpose(L, [0, 2, 1]))  # (num_ratios, input_dim, input_dim)
        elif self.quadratic_constraint_type == "symmetric_pos_diag":
            Q = tf_enforce_symmetric_and_pos_diag(unconstrained_Q, shift=5.0)
        else:
            Q = unconstrained_Q

        if self.max_spectral_norm_params:
            Q, W = self.apply_spectral_norm(Q, W, is_train)

        return Q, W, b

    def apply_spectral_norm(self, Q, W, is_train):
        print("Applying spectral normalisation to quadratic heads"
              " with params: {}".format(self.max_spectral_norm_params))

        W_sigma = tf.reduce_sum(W ** 2, axis=-1, keepdims=True) ** 0.5  # either (1, ) or (num_ratios, 1)

        if self.use_single_head:
            # Constrain spec norm to 1, since we control max norm via scales
            W = W / W_sigma

            u = tf.compat.v1.get_variable("W_power_iter_u",
                                          shape=[1, self.input_dim],
                                          initializer=initializers.TruncatedNormal(0.0, 1.0),
                                          trainable=False)
            Q = tf_spectral_norm(Q, u, training=is_train)

        else:
            max_spec_norms = tf_get_power_seq(*self.max_spectral_norm_params, self.max_num_ratios)
            max_spec_norms = tf.gather(max_spec_norms, self.bridge_idxs, axis=0)  # (num_ratios,)

            W_sigma = tf.maximum(W_sigma / tf.reshape(max_spec_norms, [-1, 1]),  1)
            W = W / W_sigma

            u = tf.compat.v1.get_variable("W_power_iter_u",
                                          shape=[self.max_num_ratios, 1, self.input_dim],
                                          initializer=initializers.TruncatedNormal(0.0, 1.0),
                                          trainable=False)
            Q = tf_batched_spectral_norm(Q, u, training=is_train, max_norms=max_spec_norms, extract_idxs=self.bridge_idxs)

        return Q, W

    @staticmethod
    def quad_energy(x, Q):
        """For each bridge, computes -x^TAx where A = L^T @ L"""
        x = tf.transpose(x, [1, 0, 2])  # (num_ratios, n, input_dim)
        y = tf.matmul(x, Q)  # (num_ratios, n, input_dim)
        e = tf.transpose(tf.reduce_sum(y * x, axis=-1))  # (n, num_ratios)
        return e  # (n, num_ratios)

    def neg_energy(self, x, is_train, is_wmark_input=False, *args):
        x = ready_x_for_per_bridge_computation(x, is_wmark_input, self.bridge_idxs)
        neg_e = self.eval(x, is_train, is_wmark_input=is_wmark_input)
        return neg_e


class BilinearHeads:
    """Final bilinear layer in a separable multi-classifier network"""

    def __init__(self,
                 input_dim,
                 bridge_idxs,
                 max_num_ratios,
                 use_single_head=False,
                 max_spectral_norm_params=None
                 ):

        self.input_dim = input_dim
        self._bridge_idxs = bridge_idxs
        self.max_num_ratios = max_num_ratios
        self.use_single_head = use_single_head
        self.max_spectral_norm_params = max_spectral_norm_params

        self.head_assignments = tf.no_op()

    @property
    def bridge_idxs(self):
        return self._bridge_idxs

    @bridge_idxs.setter
    def bridge_idxs(self, val):
        self._bridge_idxs = val

    @tf_cache_template("bilinear_heads", initializer=initializers.normal(stddev=0.005))
    def eval(self, f, g, is_train, is_wmark_input=False, *args):

        # f has shape (?, d) and g has shape (?, num_ratios, d)
        M, b = self.get_weights(is_train)
        gM = tf.matmul(tf.transpose(g, [1, 0, 2]), M)  # (num_ratios, n, input_dim)
        e = tf.transpose(tf.reduce_sum(gM * f, axis=-1))  # (n, num_ratios)
        e += b

        return -e  # (?, num_ratios)

    def get_weights(self, is_train):
        if self.use_single_head:
            M_shape = (self.input_dim, self.input_dim)
        else:
            M_shape = (self.max_num_ratios, self.input_dim, self.input_dim)

        M = tf.compat.v1.get_variable("M_all", shape=M_shape, dtype=tf.float32)
        b = tf.compat.v1.get_variable("b_all", shape=self.max_num_ratios, initializer=initializers.Zeros())

        b = tf.gather(b, self.bridge_idxs, axis=0)  # (num_ratios, )
        if not self.use_single_head:
            M = tf.gather(M, self.bridge_idxs, axis=0)  # (num_ratios, input_dim, input_dim)

        if self.max_spectral_norm_params:
            M = self.apply_spectral_norm(M, is_train)

        return M, b

    def apply_spectral_norm(self, Q, is_train):
        print("Applying spectral normalisation to bilinear heads"
              " with params: {}".format(self.max_spectral_norm_params))

        if self.use_single_head:
            # Constrain spec norm to 1, since we control max norm via scales
            u = tf.compat.v1.get_variable("W_power_iter_u",
                                          shape=[1, self.input_dim],
                                          initializer=initializers.TruncatedNormal(0.0, 1.0),
                                          trainable=False)
            Q = tf_spectral_norm(Q, u, training=is_train)

        else:
            max_spec_norms = tf_get_power_seq(*self.max_spectral_norm_params, self.max_num_ratios)
            max_spec_norms = tf.gather(max_spec_norms, self.bridge_idxs, axis=0)  # (num_ratios,)

            u = tf.compat.v1.get_variable("W_power_iter_u",
                                          shape=[self.max_num_ratios, 1, self.input_dim],
                                          initializer=initializers.TruncatedNormal(0.0, 1.0),
                                          trainable=False)
            Q = tf_batched_spectral_norm(Q, u, training=is_train, max_norms=max_spec_norms, extract_idxs=self.bridge_idxs)

        return Q

    def neg_energy(self, x, is_train, is_wmark_input=False, *args):
        raise NotImplementedError


class CondMlp:
    """Residual multi-layer perceptron with optional conditional shifts & scales at every layer"""

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 num_blocks,
                 act_name,
                 reg_coef,
                 dropout_params,
                 max_num_ratios,
                 use_cond_scale_shift,
                 use_residual=False,
                 max_spectral_norm_params=None,
                 use_final_bias=True):

        self.hidden_size = hidden_size
        self.output_size = output_size if output_size is not None else hidden_size
        self.num_blocks = num_blocks
        self.use_residual = use_residual
        self.activation = get_tf_activation(act_name)
        self.reg_coef = reg_coef
        self.dropout_final_layer = dropout_params[0]
        self.dropout_params = dropout_params[1:]
        self.max_num_ratios = max_num_ratios
        self.use_cond_scale_shift = use_cond_scale_shift
        self.max_spectral_norm_params = max_spectral_norm_params
        self.use_final_bias = use_final_bias
        self.activation_stats = []

        self.dense = SpectralDense if max_spectral_norm_params else k_layers.Dense
        self.model = self.build_mlp((input_size,))

    def build_mlp(self, shape):

        x = k_layers.Input(shape=shape)
        if self.use_cond_scale_shift:
            cond_idxs = k_layers.Input(shape=(None,), dtype=tf.int32)
        else:
            cond_idxs = None

        if self.use_residual:
            output = self.residual_blocks(x, cond_idxs)
        else:
            output = self.normal_blocks(x, cond_idxs)

        if self.use_cond_scale_shift:
            model = Model(inputs=[x, cond_idxs], outputs=output)
        else:
            model = Model(inputs=x, outputs=output)

        return model

    def residual_blocks(self, x, cond_idxs):

        h = self.dense_plus_maybe_shift_scale(x, cond_idxs, self.hidden_size, "layer0")

        for i in range(self.num_blocks):

            residual = self.activation()(h)
            residual = self.dense_plus_maybe_shift_scale(residual,
                                                         cond_idxs,
                                                         self.hidden_size,
                                                         "resblock{}_affine0".format(i),
                                                         initializer=initializers.VarianceScaling(scale=2.0))
            residual = self.activation()(residual)
            residual = self.dropout(residual, cond_idxs)
            residual = self.dense_plus_maybe_shift_scale(residual,
                                                         cond_idxs,
                                                         self.hidden_size,
                                                         "resblock{}_affine1".format(i),
                                                         initializer=initializers.VarianceScaling(scale=0.1))
            h += residual

        h = self.activation()(h)
        if self.dropout_final_layer:
            h = self.dropout(h, cond_idxs)

        h = self.dense_plus_maybe_shift_scale(h, cond_idxs, self.output_size, "final_layer")

        return h

    def normal_blocks(self, h, cond_idxs):

        for i in range(self.num_blocks):
            h = self.dense_plus_maybe_shift_scale(h, cond_idxs, self.hidden_size, "layer{}".format(i+1))
            h = self.activation()(h)
            h = self.dropout(h, cond_idxs)

        h = self.dense_plus_maybe_shift_scale(h, cond_idxs, self.hidden_size, "final_layer")

        if self.dropout_final_layer:
            h = self.dropout(h, cond_idxs)

        return h

    def dense_plus_maybe_shift_scale(self, h, cond_idxs, size, name, initializer='glorot_uniform'):
        h = self.dense(units=size,
                       kernel_initializer=initializer,
                       kernel_regularizer=tf_l2_regulariser(scale=self.reg_coef),
                       name=name)(h)

        if self.use_cond_scale_shift:
            h = CondScaleShift(self.max_num_ratios, size, name, self.max_spectral_norm_params)([h, cond_idxs])

        return h

    def dropout(self, h, cond_idxs):

        if cond_idxs is not None:
            h = CondDropout(*self.dropout_params, max_n_ratios=self.max_num_ratios, is_wmark_dim=True)([h, cond_idxs])
        else:
            h = k_layers.Dropout(self.dropout_params[0])(h)  # 'params' just contains dropout rate in this case

        return h


class CondMlpEnergy(CondMlp):
    """Residual MLP Energy function"""

    def __init__(self,
                 input_size,
                 body_hidden_size,
                 body_output_size,
                 num_blocks,
                 act_name,
                 use_residual,
                 max_spectral_norm_params,
                 reg_coef,
                 dropout_params,
                 bridge_idxs,
                 max_num_ratios,
                 use_cond_scale_shift,
                 head_type,
                 use_single_head,
                 head_multiplier,
                 quadratic_constraint_type
                 ):

        self._bridge_idxs = bridge_idxs
        self.use_cond_scale_shift = use_cond_scale_shift

        if max_spectral_norm_params is not None:
            body_spec_params = max_spectral_norm_params[1:]
            head_spec_params = max_spectral_norm_params[1:] if max_spectral_norm_params[0] else None
        else:
            body_spec_params = head_spec_params = None

        body_output_size = body_output_size if body_output_size is not None else body_hidden_size

        super().__init__(input_size=input_size,
                         hidden_size=body_hidden_size,
                         output_size=body_output_size,
                         num_blocks=num_blocks,
                         act_name=act_name,
                         reg_coef=reg_coef,
                         dropout_params=dropout_params,
                         max_num_ratios=max_num_ratios,
                         use_cond_scale_shift=use_cond_scale_shift,
                         use_residual=use_residual,
                         max_spectral_norm_params=body_spec_params)

        self.heads = init_heads(head_type=head_type,
                                body_output_size=body_output_size,
                                max_num_ratios=max_num_ratios,
                                bridge_idxs=bridge_idxs,
                                use_single_head=use_single_head,
                                max_spectral_norm_params=head_spec_params,
                                head_multiplier=head_multiplier,
                                quadratic_constraint_type=quadratic_constraint_type)

        # self.assign_new_head = self.final_layer.assign_new_head
        self.head_assignments = self.heads.head_assignments

    @property
    def bridge_idxs(self):
        return self._bridge_idxs

    @bridge_idxs.setter
    def bridge_idxs(self, val):
        self.heads.bridge_idxs = val
        self._bridge_idxs = val

    def neg_energy(self, x, is_train, is_wmark_input=False):

        x = collapse_event_dims(x, is_wmark_input)

        if self.use_cond_scale_shift:
            x = ready_x_for_per_bridge_computation(x, is_wmark_input, self.bridge_idxs)  # (?, n_ratios, *event_dims)
            h = self.model([x, tf.convert_to_tensor(self.bridge_idxs)], training=is_train)
        else:
            h = self.model([x, tf.convert_to_tensor(self.bridge_idxs)], training=is_train)
            h = ready_x_for_per_bridge_computation(h, is_wmark_input, self.bridge_idxs)  # (?, n_ratios, output_size)

        neg_e = self.heads.eval(h, is_train, is_wmark_input=is_wmark_input)

        return neg_e  # (n, num_ratios) if is_train==True else (2n, num_ratios)


class ResNet:

    def __init__(self,
                 channel_widths,
                 dense_hidden_size,
                 act_name,
                 reg_coef,
                 dropout_params,
                 max_num_ratios,
                 use_cond_scale_shift,
                 shift_scale_per_channel,
                 use_instance_norm,
                 max_spectral_norm_params,
                 just_track_spectral_norm,
                 img_shape,
                 use_average_pooling,
                 use_global_sum_pooling,
                 use_attention,
                 final_pool_shape=(2, 2),
                 kernel_shape=(3, 3),
                 init_kernel_shape=(3, 3),
                 init_kernel_strides=(1, 1),
                 **kwargs
                 ):

        self.channel_widths = channel_widths
        self.final_pool_shape = final_pool_shape
        self.init_kernel_shape = init_kernel_shape
        self.init_kernel_strides = init_kernel_strides

        self.kernel_shape = kernel_shape
        self.output_size = dense_hidden_size
        self.activation = get_tf_activation(act_name)
        self.reg_coef = reg_coef
        self.dropout_final_layer = dropout_params[0]
        self.dropout_params = dropout_params[1:]
        self.max_num_ratios = max_num_ratios

        self.use_cond_scale_shift = use_cond_scale_shift
        self.shift_scale_per_channel = shift_scale_per_channel
        self.use_instance_norm = use_instance_norm
        self.use_attention = use_attention
        if max_spectral_norm_params is not None:
            self.max_spectral_norm_params = max_spectral_norm_params[1:]
        else:
            self.max_spectral_norm_params = None
        self.just_track_spectral_norm = just_track_spectral_norm

        self.img_shape = img_shape
        self.use_average_pooling = use_average_pooling
        self.use_global_sum_pooling = use_global_sum_pooling

        if max_spectral_norm_params or just_track_spectral_norm:
            self.conv_fn = SpectralConv
            self.dense_fn = SpectralDense
        else:
            self.conv_fn = make_flexible_Conv2D_fn()
            self.dense_fn = make_flexible_Dense_fn()

        self.model = self.build_resnet(self.channel_widths, self.img_shape)

        for loss in self.model.losses:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, loss)

    def build_resnet(self, channel_widths, shape):

        inputs = k_layers.Input(shape=shape)
        if self.use_cond_scale_shift:
            cond_idxs = k_layers.Input(shape=(), dtype=tf.int32)
        else:
            cond_idxs = None

        x = self.init_layer(channel_widths, cond_idxs, inputs, shape)
        x = self.res_blocks(channel_widths, self.kernel_shape, cond_idxs, x)
        x = self.activation()(x)

        reduce = k_layers.Lambda(lambda a: tf.reduce_sum(a, axis=[1, 2]))
        x = reduce(x)  # global sum pooling

        x = self.dense_and_shift_scale(x, cond_idxs, self.output_size, name='final_layer')
        x = self.activation()(x)
        if self.dropout_final_layer:
            x = self.dropout(x, cond_idxs)

        if self.use_cond_scale_shift:
            model = Model(inputs=[inputs, cond_idxs], outputs=x)
        else:
            model = Model(inputs=inputs, outputs=x)

        return model

    def init_layer(self, channel_widths, cond_idxs, inputs, shape):

        x = self.conv_fn(channel_widths[0][0],
                         self.init_kernel_shape,
                         strides=self.init_kernel_strides,
                         padding='SAME',
                         input_shape=shape,
                         kernel_regularizer=regularizers.l2(self.reg_coef),
                         name='conv2d',
                         use_bias=(cond_idxs is None),
                         just_track_spectral_norm=self.just_track_spectral_norm)(inputs)

        if cond_idxs is not None:
            x = CondConvScaleShift(self.max_num_ratios, self.shift_scale_per_channel,
                                   self.use_instance_norm, self.max_spectral_norm_params)([x, cond_idxs])

        return x

    def res_blocks(self, channel_widths, kernel_shape, cond_idxs, x):

        block_counter = 0
        for i, widths in enumerate(channel_widths[1:]):
            for j, width in enumerate(widths):

                name = 'res_block_{}'.format(block_counter)
                use_adaptive = (j == 0) and (width != channel_widths[i - 1][-1])

                x = self.res_block(x, width, kernel_shape, name=name, pool_shape=(2, 2),
                                   use_pooling=(j == 0), adaptive=use_adaptive, cond_idxs=cond_idxs)
                block_counter += 1

                if block_counter == 1 and self.use_attention:
                    x = self.attention(x, channel_widths[0][0])

        return x

    def res_block(self, x, n_channels, kernel_shape, name, pool_shape=(2, 2), use_pooling=False, adaptive=False,
                  cond_idxs=None):
        """ act - conv - condshiftscale - act - conv - condshiftscale - average pool - add residual"""

        res = x
        res = self.activation()(res)
        res = self.conv_and_shift_scale(res, n_channels, kernel_shape, name + "_conv1", cond_idxs=cond_idxs)
        res = self.dropout(res, cond_idxs)
        res = self.activation()(res)
        res = self.conv_and_shift_scale(res, n_channels, kernel_shape, name + "_conv2", cond_idxs=cond_idxs)
        # zero_init=True breaks when used in conjuction with spectral norm

        if use_pooling:
            if self.use_average_pooling:
                res = k_layers.AveragePooling2D(pool_size=pool_shape, padding='SAME')(res)
            else:
                res = k_layers.MaxPooling2D(pool_size=pool_shape, padding='SAME')(res)

        if adaptive:
            x = self.conv_and_shift_scale(x, n_channels, (1, 1), name + "_conv3", cond_idxs=cond_idxs)

        if use_pooling:
            if self.use_average_pooling:
                x = k_layers.AveragePooling2D(pool_size=pool_shape, padding='SAME')(x)
            else:
                x = k_layers.MaxPooling2D(pool_size=pool_shape, padding='SAME')(x)

        x = k_layers.Add()([x, res])

        return x

    def conv_and_shift_scale(self, x, n_channels, kernel_shape, name, cond_idxs=None, zero_init=False):

        kernel_init = 'zeros' if zero_init else 'glorot_uniform'
        x = self.conv_fn(n_channels,
                         kernel_shape,
                         padding='SAME',
                         name=name,
                         kernel_initializer=kernel_init,
                         kernel_regularizer=regularizers.l2(self.reg_coef),
                         use_bias=(cond_idxs is None),
                         just_track_spectral_norm=self.just_track_spectral_norm)(x)

        if cond_idxs is not None:
            x = CondConvScaleShift(self.max_num_ratios, self.shift_scale_per_channel,
                                   self.use_instance_norm, self.max_spectral_norm_params)([x, cond_idxs])

        return x

    def dropout(self, x, cond_idxs):

        if cond_idxs is not None:
            x = CondDropout(*self.dropout_params, max_n_ratios=self.max_num_ratios, is_wmark_dim=False)([x, cond_idxs])
        else:
            x = k_layers.Dropout(self.dropout_params[0])(x)  # 'params' just contains dropout rate in this case

        return x

    def dense_and_shift_scale(self, h, cond_idxs, size, name):

        reg = regularizers.l2(self.reg_coef)
        h = self.dense_fn(units=size, kernel_regularizer=reg, name=name,
                          just_track_spectral_norm=self.just_track_spectral_norm)(h)

        if self.use_cond_scale_shift:
            h = CondScaleShift(self.max_num_ratios, size, name, self.max_spectral_norm_params)([h, cond_idxs])

        return h

    @staticmethod
    def attention(x, channels):

        f = k_layers.Conv2D(channels // 8, (1, 1), padding='SAME', name='f_attn_conv')(x)  # [bs, h, w, c']
        g = k_layers.Conv2D(channels // 8, (1, 1), padding='SAME', name='g_attn_conv')(x)  # [bs, h, w, c']
        h = k_layers.Conv2D(channels, (1, 1), padding='SAME', name='h_attn_conv')(x)  # [bs, h, w, c']

        # N = h * w
        s = tf_keras_matmul(tf_keras_hw_flatten(g), tf_keras_hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = k_layers.Softmax()(s)  # attention map

        o = tf_keras_matmul(beta, tf_keras_hw_flatten(h))  # [bs, N, C]
        o = k_layers.Reshape(x.shape[1:])(o)  # [bs, h, w, C]
        o = k_layers.Conv2D(channels, (1, 1), padding='SAME', name='attn_conv')(o)
        o = ScaleShift(1, "attn_gamma", use_shift=False, scale_init='zeros')(o)

        x = o + x

        return x


class ResNetEnergy(ResNet):
    """Resnet energy function"""

    def __init__(self, **kwargs):

        self._bridge_idxs = kwargs["bridge_idxs"]
        self.num_ratios = shape_list(self._bridge_idxs)[0]

        super().__init__(**kwargs)

        if (kwargs["max_spectral_norm_params"] is not None) and kwargs["max_spectral_norm_params"][0]:
            head_spec_params = kwargs["max_spectral_norm_params"][1:]
        else:
            head_spec_params = None
        del kwargs["max_spectral_norm_params"]

        self.heads = init_heads(body_output_size=self.output_size,
                                max_spectral_norm_params=head_spec_params,
                                **kwargs
                                )

        # self.assign_new_head = self.final_layer.assign_new_head
        self.head_assignments = self.heads.head_assignments

    @property
    def bridge_idxs(self):
        return self._bridge_idxs

    @bridge_idxs.setter
    def bridge_idxs(self, val):
        self.heads.bridge_idxs = val
        self._bridge_idxs = val
        self.num_ratios = shape_list(self._bridge_idxs)[0]

    def neg_energy(self, x, is_train, is_wmark_input=False):

        x_shp = shape_list(x)

        if self.use_cond_scale_shift:
            x, cond_idxs = ready_x_for_per_bridge_convnet(x, is_wmark_input, self.bridge_idxs)
            h = self.model([x, cond_idxs], training=is_train)
            h = tf.reshape(h, [-1, self.num_ratios, self.output_size])  # (?, num_ratios, out_dim)
            neg_e = self.heads.eval(h, is_train=is_train, is_wmark_input=is_wmark_input)  # (?, num_ratios)
        else:
            if is_wmark_input: x = tf.reshape(x, [-1, *x_shp[2:]])  # (n*n_waymarks, ...)
            h = self.model(x, training=is_train)
            if is_wmark_input: h = tf.reshape(h, [-1, self.num_ratios + 1, self.output_size])  # (n, n_waymarks, out_dim)
            h = ready_x_for_per_bridge_computation(h, is_wmark_input, self.bridge_idxs)
            neg_e = self.heads.eval(h, is_train=is_train, is_wmark_input=is_wmark_input)  # (?, num_ratios)

        return neg_e  # (?, num_ratios)


class SeparableEnergy:

    def __init__(self, bridge_idxs, max_num_ratios, config, only_f=False):

        # note: this code duplicates code from elsewhere - needs refactoring eventually

        self._bridge_idxs = bridge_idxs
        if bridge_idxs is not None:
            self.num_ratios = shape_list(self.bridge_idxs)[0]
        self.max_num_ratios = max_num_ratios
        self.use_cond_scale_shift = config.use_cond_scale_shift
        self.network_type = config.network_type
        self.only_f = only_f

        self.build_nets(config)
        self.output_size = self.f.output_size

        if not only_f:
            self.heads = BilinearHeads(
                input_dim=self.output_size,
                bridge_idxs=bridge_idxs,
                max_num_ratios=max_num_ratios,
                use_single_head=config.get("use_single_head", False),
                max_spectral_norm_params=config.get("max_spectral_norm_params", None)
            )

    @property
    def bridge_idxs(self):
        return self._bridge_idxs

    @bridge_idxs.setter
    def bridge_idxs(self, val):
        self.heads.bridge_idxs = val
        self._bridge_idxs = val
        self.num_ratios = shape_list(self._bridge_idxs)[0]

    def neg_energy(self, xy, is_train, is_wmark_input=False):

        assert not self.only_f, "'only_f=True' was passed into the constructor, which means that the g network" \
                                "was not built, and thus the log-ratio cannot be computed"

        x, y = xy  # x.shape = (n_batch, *event_dims), y.shape = (n_batch, n_waymarks, *event_dims)

        # compute the final hidden vectors for each x & y_k
        f_x = self.compute_f_hiddens(x, is_train)
        g_y = self.compute_g_hiddens(y, is_train, is_wmark_input)  # (?, n_ratios, output_size)

        # duplicate fx along axis 0
        # (by default, the duplication factor will be 2 to account for positive & negative samples)
        dup_factor = tf.cast(shape_list(g_y)[0] / shape_list(f_x)[0], dtype=tf.int32)
        f_x = tf.tile(f_x, [dup_factor, 1])

        # map these vectors to a scalar
        neg_e = self.heads.eval(f_x, g_y, is_train=is_train, is_wmark_input=is_wmark_input)

        return neg_e  # (?, num_ratios)

    def compute_f_hiddens(self, x, is_train):
        if self.network_type == "mlp":
            x = collapse_event_dims(x, is_wmark_input=False)
        return self.f.model(x, training=is_train)  # (?, output_size)

    def compute_g_hiddens(self, y, is_train, is_wmark_input):

        if self.network_type == "mlp":
            y = collapse_event_dims(y, is_wmark_input=is_wmark_input)

        y_shp = shape_list(y)
        if self.network_type == "resnet":
            if self.use_cond_scale_shift:
                y, cond_idxs = ready_x_for_per_bridge_convnet(y, is_wmark_input, self.bridge_idxs)
                h = self.g.model([y, cond_idxs], training=is_train)
                h = tf.reshape(h, [-1, self.num_ratios, self.output_size])  # (?, num_ratios, out_dim)
            else:
                if is_wmark_input: y = tf.reshape(y, [-1, *y_shp[2:]])  # (n*n_waymarks, ...)
                h = self.g.model(y, training=is_train)
                if is_wmark_input: h = tf.reshape(h, [-1, self.num_ratios + 1, self.output_size])  # (n, n_waymarks, out_dim)
                h = ready_x_for_per_bridge_computation(h, is_wmark_input, self.bridge_idxs)  # (?, n_ratios, d)

        elif self.network_type == "mlp":
            if self.use_cond_scale_shift:
                y = ready_x_for_per_bridge_computation(y, is_wmark_input, self.bridge_idxs)  # (?, n_ratios, d)
                h = self.g.model([y, tf.convert_to_tensor(self.bridge_idxs)], training=is_train)
            else:
                h = self.g.model([y, tf.convert_to_tensor(self.bridge_idxs)], training=is_train)
                h = ready_x_for_per_bridge_computation(h, is_wmark_input, self.bridge_idxs)  # (?, n_ratios, d)

        return h  # (?, n_ratios, output_size)


    def build_nets(self, config):

        if config.network_type == "resnet":
            if not self.only_f:
                with tf.variable_scope("g_network"):
                    self.g = ResNet(channel_widths=config.channel_widths,
                                    dense_hidden_size=config.mlp_hidden_size,
                                    act_name=config.activation_name,
                                    reg_coef=config.energy_reg_coef,
                                    dropout_params=config.dropout_params,
                                    max_num_ratios=self.max_num_ratios,
                                    use_cond_scale_shift=config.use_cond_scale_shift,
                                    shift_scale_per_channel=config.shift_scale_per_channel,
                                    use_instance_norm=config.use_instance_norm,
                                    max_spectral_norm_params=config.get("max_spectral_norm_params", None),
                                    just_track_spectral_norm=config.get("just_track_spectral_norm", False),
                                    img_shape=config.data_args["img_shape"][:-1],
                                    use_average_pooling=config.get("use_average_pooling", True),
                                    use_global_sum_pooling=config.use_global_sum_pooling,
                                    use_attention=config.use_attention,
                                    final_pool_shape=config.final_pool_shape,
                                    kernel_shape=config.conv_kernel_shape,
                                    init_kernel_shape=config.get("init_kernel_shape", (3, 3)),
                                    init_kernel_strides=config.get("init_kernel_strides", (1, 1))
                                    )
            with tf.variable_scope("f_network"):
                self.f = ResNet(channel_widths=config.channel_widths,
                                dense_hidden_size=config.mlp_hidden_size,
                                act_name=config.activation_name,
                                reg_coef=config.energy_reg_coef,
                                dropout_params=config.dropout_params,
                                max_num_ratios=None,
                                use_cond_scale_shift=False,
                                shift_scale_per_channel=False,
                                use_instance_norm=False,
                                max_spectral_norm_params=config.get("max_spectral_norm_params", None),
                                just_track_spectral_norm=config.get("just_track_spectral_norm", False),
                                img_shape=config.data_args["img_shape"][:-1],
                                use_average_pooling=config.get("use_average_pooling", True),
                                use_global_sum_pooling=config.use_global_sum_pooling,
                                use_attention=config.use_attention,
                                final_pool_shape=config.final_pool_shape,
                                kernel_shape=config.conv_kernel_shape,
                                init_kernel_shape=config.get("init_kernel_shape", (3, 3)),
                                init_kernel_strides=config.get("init_kernel_strides", (1, 1))
                                )

        elif config.network_type == "mlp":
            if not self.only_f:
                with tf.variable_scope("g_network"):
                    self.g = CondMlp(input_size=int(config.n_dims/2),
                                     hidden_size=config.mlp_hidden_size,
                                     output_size=config.mlp_output_size,
                                     num_blocks=config.mlp_n_blocks,
                                     act_name=config.activation_name,
                                     reg_coef=config.energy_reg_coef,
                                     dropout_params=config.dropout_params,
                                     max_num_ratios=self.max_num_ratios,
                                     use_cond_scale_shift=config.use_cond_scale_shift,
                                     use_residual=config.use_residual_mlp,
                                     max_spectral_norm_params=config.get("max_spectral_norm_params", None)
                                     )
            with tf.variable_scope("f_network"):
                self.f = CondMlp(input_size=int(config.n_dims/2),
                                 hidden_size=config.mlp_hidden_size,
                                 output_size=config.mlp_output_size,
                                 num_blocks=config.mlp_n_blocks,
                                 act_name=config.activation_name,
                                 reg_coef=config.energy_reg_coef,
                                 dropout_params=config.dropout_params,
                                 max_num_ratios=None,
                                 use_cond_scale_shift=False,
                                 use_residual=config.use_residual_mlp,
                                 max_spectral_norm_params=config.get("max_spectral_norm_params", None)
                                 )


class GaussEnergy:
    """Unnormalised gaussian with identity covariance matrix"""

    def __init__(self, n_dims, normalised=False):
        self.n_dims = n_dims
        self.c = 0.  # scaling param (approximates log partition)
        self.true_partition = tf.constant(0., tf.float32)
        self.normalised = normalised

    @tf_cache_template("mlp_energy", initializer=initializers.glorot_normal())
    def log_prob(self, x):
        if self.normalised:
            loc = tf.compat.v1.get_variable("gauss_loc", shape=self.n_dims, dtype=tf.float32, initializer=tf.zeros_initializer())
            scale = tf.compat.v1.get_variable("gauss_scale", dtype=tf.float32, initializer=tf.eye(self.n_dims))
            ebm = tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=scale)
            energy = -ebm.log_prob(x)
        else:
            # use diagonal covariance, so partition function is simple
            mean = tf.compat.v1.get_variable("gauss_mean", self.n_dims, dtype=tf.float32)
            diag = tf.compat.v1.get_variable("gauss_diag", self.n_dims, dtype=tf.float32)
            self.true_partition = 0.5 * (self.n_dims * tf.log(2 * np.pi) - tf.reduce_sum(diag))
            x_centred = x - mean
            energy = 0.5 * tf.reduce_sum(tf.square(x_centred) * tf.exp(diag), axis=1)

        self.c = tf.compat.v1.get_variable("scaling_param", (), dtype=tf.float32, initializer=tf.zeros_initializer())
        return tf.squeeze(-energy - self.c)


class ProductModel:

    def __init__(self, energy_fn, noise_dist):
        self.energy_fn = energy_fn
        self.noise_dist = noise_dist
        self.scaling_params = energy_fn.c
        self.c = tf.reduce_sum(self.scaling_params)

    def log_prob(self, x):
        """Return (unnormalised) log prob"""
        noise_log_p = self.noise_dist.log_prob(x)  # (batch_size, )
        log_ratios = self.energy_fn.neg_energy(x, is_training=False)  # (batch_size, n_ratios)
        self.vals_per_energy = tf.concat([log_ratios, tf.expand_dims(noise_log_p, -1)], axis=-1)
        self.av_val_per_energy = tf.reduce_mean(log_ratios, axis=0)  # (n_ratios, )
        sum_log_ratios = tf.reduce_sum(log_ratios, axis=-1)  # (batch_size, )

        return noise_log_p + sum_log_ratios  # (batch_size, )


def ready_x_for_per_bridge_computation(x, is_waymark_input, bridge_idxs):
    if is_waymark_input:
        x = tf_duplicate_waymarks(x)  # (2n, n_ratios, d)
    else:
        num_ratios = shape_list(bridge_idxs)[0]
        event_dims = shape_list(x)[1:]
        x = tf.tile(tf.expand_dims(x, axis=1), [1, num_ratios] + [1]*len(event_dims))  # (n, n_ratios, *event_dims)

    return x


def ready_x_for_per_bridge_convnet(x, is_train, bridge_idxs):
    x_shp = shape_list(x)
    if is_train:
        n, k, event_dims = x_shp[0], x_shp[1], x_shp[2:]

        # just use standard TRE loss, with no extra alignment terms
        x = tf_duplicate_waymarks(x, concat=False)  # (2, n, num_ratios, ...)
        x = tf.reshape(x, [-1, *event_dims])  # (2*n*num_ratios, ...)

        cond_idxs = tf.tile(tf.reshape(bridge_idxs, [1, 1, -1]), [x_shp[0], 2, 1])  # (n, 2, n_ratios)
        cond_idxs = tf.reshape(cond_idxs, [-1])  # (n*2*n_ratios, )
    else:
        num_ratios = shape_list(bridge_idxs)[0]
        x = tf_repeat_first_axis(x, num_ratios)  # (n*n_ratios, ...)
        cond_idxs = tf.tile(bridge_idxs, [x_shp[0]])  # (n*n_ratios, )

    return x, cond_idxs


def collapse_event_dims(x, is_wmark_input):
    x_shp = shape_list(x)
    if is_wmark_input:
        x = tf.reshape(x, x_shp[:2] + [-1])
    else:
        x = tf.reshape(x, x_shp[:1] + [-1])
    return x


def init_heads(head_type,
               body_output_size,
               max_num_ratios,
               bridge_idxs,
               use_single_head,
               max_spectral_norm_params,
               head_multiplier,
               quadratic_constraint_type,
               **kwargs
               ):

    if head_type == "linear":
        heads = LinearHeads(input_dim=body_output_size,
                            bridge_idxs=bridge_idxs,
                            max_num_ratios=max_num_ratios,
                            head_multiplier=head_multiplier,
                            max_spectral_norm_params=max_spectral_norm_params
                            )

    elif head_type == "quadratic":
        heads = QuadraticHeads(input_dim=body_output_size,
                               bridge_idxs=bridge_idxs,
                               max_num_ratios=max_num_ratios,
                               use_single_head=use_single_head,
                               max_spectral_norm_params=max_spectral_norm_params,
                               quadratic_constraint_type=quadratic_constraint_type)

    else:
        raise ValueError("Final layer of type '{}' not recognised. "
                         "'linear' or 'quadratic' are currently valid options".format(head_type))

    return heads
