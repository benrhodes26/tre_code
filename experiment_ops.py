import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from models import *
from utils.misc_utils import *
from utils.tf_utils import *
from utils.experiment_utils import *
from utils.plot_utils import *


class TFCorrelatedGaussians:

    def __init__(self, n_dims, correlation_coefficient):
        self.n_dims = n_dims
        self.half_dims = tf.cast(self.n_dims/2, tf.int32)
        rho = correlation_coefficient
        self.dist = tfd.MultivariateNormalFullCovariance(loc=tf.zeros(2, tf.float32),
                                                         covariance_matrix=tf.constant(
                                                             np.array([[1.0, rho], [rho, 1.0]]),
                                                             dtype=tf.float32),
                                                         )

    def log_prob(self, x):
        x_reshaped = tf.reshape(x, [-1, self.half_dims, 2])
        log_p_reshaped = self.dist.log_prob(x_reshaped)  # (n, d/2)
        return tf.reduce_sum(log_p_reshaped, axis=1)

    def sample(self, sample_shape):
        x = self.dist.sample((sample_shape, self.half_dims))  # (n, d/2, 2)
        return tf.reshape(x, [sample_shape, self.n_dims])


class CustomMixture:

    def __init__(self, dist1, dist2, weight1):
        self.w1 = weight1
        self.w2 = 1 - weight1
        self.dist1 = dist1
        self.dist2 = dist2

    def log_prob(self, x):
        logits = tf.stack([self.dist1.log_prob(x) + tf.log(self.w1),
                           self.dist2.log_prob(x) + tf.log(self.w2)], axis=1)
        return tf.reduce_logsumexp(logits, axis=1)

    def sample(self, sample_shape):
        u = tfd.Uniform(0, 1).sample(sample_shape)
        s = tf.where(u <= self.w1, self.dist1.sample(sample_shape), self.dist2.sample(sample_shape))
        return s


class CustomMixtureWithInvertibleComponent(CustomMixture):

    class CustomBaseDist:

        def __init__(self, dist1, dist2, weight1):
            self.dist1 = dist1  # the invertible distribution
            self.dist2 = dist2  # the other distribution
            self.w1 = weight1

        def sample(self, shape):
            u = tfd.Uniform(0, 1).sample(shape)

            dist1_inv_sample = self.dist1.base_dist.sample(shape)
            dist2_sample = self.dist2.sample(shape)
            dist2_inv_sample = self.dist1.inverse(dist2_sample)

            return tf.where(u <= self.w1, dist1_inv_sample, dist2_inv_sample)

        def log_prob(self, z):
            beta_1 = tf.log(self.w1) + self.dist1.base_dist.log_prob(z)  # (n, )

            x, ldj = self.dist1.forward(z, ret_ldj=True)
            beta_2 = tf.log(1 - self.w1) + self.dist2.log_prob(x) + ldj  # (n, )

            logits = tf.stack([beta_1, beta_2], axis=1)
            return tf.reduce_logsumexp(logits, axis=1)

    def __init__(self, dist1, dist2, weight1):
        self.base_dist = self.CustomBaseDist(dist1, dist2, weight1)
        super(CustomMixtureWithInvertibleComponent, self).__init__(dist1, dist2, weight1)

    def inverse(self, x, **kwargs):
        return self.dist1.inverse(x, **kwargs)

    def forward(self, z, **kwargs):
        return self.dist1.forward(z, **kwargs)

    def sample_base_dist(self, shape):
        return self.base_dist.sample(shape)


def build_noise_dist(name, data, config, event_shape=None, flow_training_bool=None):

    if event_shape is None:
        event_shape = shape_list(data)[1:]
    event_dims_rank = len(event_shape) if isinstance(event_shape, list) else 1

    if "gaussian" in name:
        if ("noise_dist_gaussian_loc" not in config) or config.noise_dist_gaussian_loc is None:
            loc = tf.zeros(event_shape, tf.float32)
        else:
            loc = tf.ones(event_shape, tf.float32) * config.noise_dist_gaussian_loc

        if ("noise_dist_gaussian_std" not in config) or config.noise_dist_gaussian_std is None:
            gaussian_stds = tf.ones(shape=event_shape, dtype=tf.float32)
        else:
            gaussian_stds = tf.ones(event_shape, tf.float32) * config.noise_dist_gaussian_std

        with tf.compat.v1.variable_scope("noise_dist"):
            if "full_covariance" in name:
                cov = tf.convert_to_tensor(config.cov_mat.astype(np.float32), dtype=tf.float32)
                pre_noise_dist = tfd.MultivariateNormalFullCovariance(loc=tf.zeros(np.prod(event_shape)), covariance_matrix=cov)
                noise_dist = tfd.TransformedDistribution(distribution=pre_noise_dist, bijector=tfp.bijectors.Reshape(event_shape))
            else:
                noise_dist = tfd.Independent(tfd.Normal(name="noise_dist",
                                                        loc=loc,
                                                        scale=gaussian_stds),
                                             reinterpreted_batch_ndims=event_dims_rank)

    elif name == "flow":
        noise_dist, _ = build_flow(config, data, flow_training_bool=flow_training_bool)

    else:
        raise ValueError("name of noise distribution must contain 'uniform', 'marginals', 'gaussian' "
                         ", 'full_covariance_gaussian' or 'flow'")

    return noise_dist


def build_data_dist(name, conf, data=None, correlation_coefficient=None):
    with tf.variable_scope("data_dist"):
        if name == "gaussian":
            data_dist = build_blockwise_correlated_gaussians(conf.n_dims, correlation_coefficient)
        elif name == "flow":
            data_dist, _ = build_flow(conf, data)
        else:
            raise ValueError("name of target distribution can only be 'gaussian' or 'flow'."
                             " '{}' is not a valid option.".format(name))
    return data_dist


def build_blockwise_correlated_gaussians(n_dims, rho):
    cov_mat = tf.eye(2, dtype=tf.float32)
    cov_mat += tf.scatter_nd(indices=[[0, 1], [1, 0]], updates=[rho, rho], shape=[2, 2])
    dist = tfd.Independent(
            tfd.MultivariateNormalFullCovariance(
                loc=[tf.zeros(2, tf.float32) for _ in range(int(n_dims/2))],
                covariance_matrix=[cov_mat for _ in range(int(n_dims/2))]
            ),
            reinterpreted_batch_ndims=1
    )

    return dist


def build_blockwise_correlated_gaussian_waymarks(n_dims, n_waymarks, rhos):

    locs = tf.tile(tf.zeros((1, 1, 2), tf.float32), [n_waymarks, int(n_dims/2), 1])
    cov_mat = tf.eye(2, dtype=tf.float32)

    def cond(i, _):
        return i < n_waymarks

    def body(i, covs):
        cov = cov_mat + tf.scatter_nd([[0, 1], [1, 0]], [rhos[i], rhos[i]], [2, 2])
        cov = tf.tile(tf.expand_dims(cov, axis=0), [int(n_dims/2), 1, 1])
        covs = covs.write(i, cov)
        return i+1, covs

    i0 = tf.constant(0)
    covs0 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    _, covs = tf.while_loop(cond, body, loop_vars=[i0, covs0])
    covs = covs.stack()
    print("covs shape:", shape_list(covs))

    dist = tfd.Independent(
        tfd.MultivariateNormalFullCovariance(loc=locs, covariance_matrix=covs),
        reinterpreted_batch_ndims=1
    )

    return dist


def sample_noise_dist(sess, graph, noise_dist_name, dp, n_samples = None):
    if noise_dist_name == "marginals":
        noise_samples = seperately_permute_matrix_cols(dp.data)
    else:
        if n_samples is None: n_samples = dp.data.shape[0]

        def feed_dict_fn(j, n, b):
            return {graph.n_noise_samples: min(b, n-j)}

        noise_samples = tf_batched_operation(sess=sess,
                                             ops=graph.noise_samples,
                                             n_samples=n_samples,
                                             batch_size=min(1000, n_samples),
                                             feed_dict_fn=feed_dict_fn
                                             )
    return noise_samples


def build_flow(config, data, flow_training_bool=None, flow_keep_prob=1.0, flow_reg_coef=0.0):

    if flow_training_bool is None:
        flow_training_bool = tf.placeholder_with_default(False, shape=(), name="flow_training_bool")

    no_logit = ("logit" not in config.data_args) or (not config.data_args["logit"])
    if config.dataset_name in ["mnist"] and no_logit:
        logit_alpha = 1e-6
    else:
        logit_alpha = None

    with tf.variable_scope("flow"):
        flow = Flow(input_dim=config.n_dims,
                    num_bijectors=config.flow_n_bijectors,
                    n_layers_or_blocks=config.flow_num_layers_or_blocks,
                    hidden_size=config.flow_hidden_size,
                    activation_name=config.get("flow_activation", "relu"),
                    training=flow_training_bool,
                    n_mixture_components=config.mogmade_n_mixture_components,
                    flow_type=config.flow_type,
                    use_batchnorm=False,
                    # use_batchnorm=True,
                    dropout_keep_p=flow_keep_prob,
                    reg_coef=flow_reg_coef,
                    seed=None,
                    init_data=data,
                    img_shape=config.data_args.get("img_shape", None),
                    glow_depth=config.glow_depth if "glow_depth" in config else 8,
                    glow_use_split=config.glow_use_split if "glow_use_split" in config else True,
                    glow_coupling_type=config.glow_coupling_type if "glow_coupling_type" in config else "rational_quadratic",
                    flow_num_spline_bins=config.get("flow_num_spline_bins", 8),
                    glow_temperature=config.glow_temperature,
                    num_splines=config.num_splines,
                    spline_interval_min=config.spline_interval_min,
                    nbins_for_splines=config.nbins_for_splines,
                    logit_copula_marginals=config.logit_copula_marginals,
                    data_minmax=config.train_data_min_max,
                    logit_alpha=logit_alpha,
                    preprocess_shift=config.get("preprocess_shift", 0.0),
                    preprocess_logit_shift=config.get("preprocess_logit_shift", None),
                    per_dim_stds=config.get("train_data_stds", None) if config.dataset_name == "mnist" else None
                    )
        flow_log_prob = flow.log_prob(data)

    return flow, flow_log_prob


# noinspection PyUnresolvedReferences
def plot_chains(chains, name, save_dir, dp, config, graph=None, sess=None,
                rank_op=None, plot_hists=False, is_annealed_samples=False, ret_chains=False):

    n, k = chains.shape[0], chains.shape[1]

    if plot_hists:
        for j in range(min(5, k)):
            plot_hists_for_each_dim(n_dims_to_plot=10,
                                    data=chains[:, j, ...].reshape(n, -1),
                                    dir_name=save_dir + "hists_and_scatters/",
                                    filename="{}_state_{}_along_chain".format(name, j)
                                    )

    if "2d" in config.dataset_name or "1d" in config.dataset_name:
        dp, _ = load_data_providers_and_update_conf(config)
        dp.source_1d_or_2d.plot_sequences(
            data=chains, dir_name=save_dir, s=0.1,
            name="{}_sampled_waymarks".format(name) if is_annealed_samples else name,
            label_type="sampled_waymarks" if is_annealed_samples else None)
        dp.source_1d_or_2d.plot_sequences(chains[:, -1:, :], save_dir, "{}_final_post_annealed_samples".format(name), s=0.1)

    # rank image samples by log-density and plot them
    if rank_op is not None:
        rank_metric = tf_batched_operation(sess=sess,
                                           ops=rank_op,
                                           n_samples=chains.shape[0],
                                           batch_size=config.n_batch,
                                           data_pholder=graph.data,
                                           data=chains[:, -1, ...])
        sort_idxs = np.argsort(rank_metric)
        chains = chains[sort_idxs]

    name = name + "_low_to_high_loglik"
    plot_chains_main(chains, name, save_dir, dp, config)

    if ret_chains:
        return chains, sort_idxs


def build_energies(config,
                   bridge_idxs,
                   max_num_ratios,
                   head_multiplier=1.0,
                   eval_only_f=False  # only relevant for MI estimation when separable network is used
                   ):

    if not config.do_mutual_info_estimation:

        if config.network_type == "linear":
            energy_obj = LinearHeads(input_dim=config.n_dims,
                                     bridge_idxs=bridge_idxs,
                                     max_num_ratios=max_num_ratios,
                                     use_single_head=config.get("use_single_head", False),
                                     max_spectral_norm_params=config.get("max_spectral_norm_params", None)
                                     )

        elif config.network_type == "quadratic":
            energy_obj = QuadraticHeads(input_dim=config.n_dims,
                                        bridge_idxs=bridge_idxs,
                                        max_num_ratios=max_num_ratios,
                                        use_single_head=config.get("use_single_head", False),
                                        max_spectral_norm_params=config.get("max_spectral_norm_params", None),
                                        quadratic_constraint_type=config.get("quadratic_constraint_type", "semi_pos_def"),
                                        use_linear_term=config.get("quadratic_head_use_linear_term", True),
                                        reg_coef=config.get("quadratic_head_reg_coef", 0.)
                                        )

        elif config.network_type == "resnet":
            if config.dataset_name not in IMG_DATASETS:
                raise ValueError("Must include {} inside IMG_DATASETS in project_constants.py".format(config.dataset_name))
            elif config.data_args["img_shape"] is None:
                raise ValueError("Must specify an img_shape inside the data_args dict within config file")

            energy_obj = ResNetEnergy(channel_widths=config.channel_widths,
                                      dense_hidden_size=config.mlp_hidden_size,
                                      act_name=config.activation_name,
                                      reg_coef=config.energy_reg_coef,
                                      dropout_params=config.dropout_params,
                                      bridge_idxs=bridge_idxs,
                                      max_num_ratios=max_num_ratios,
                                      head_type=config.head_type,
                                      use_single_head=config.get("use_single_head", False),
                                      use_cond_scale_shift=config.use_cond_scale_shift,
                                      shift_scale_per_channel=config.shift_scale_per_channel,
                                      use_instance_norm=config.use_instance_norm,
                                      max_spectral_norm_params=config.get("max_spectral_norm_params", None),
                                      just_track_spectral_norm=config.get("just_track_spectral_norm", False),
                                      img_shape=config.data_args["img_shape"],
                                      use_average_pooling=config.get("use_average_pooling", True),
                                      use_global_sum_pooling=config.use_global_sum_pooling,
                                      use_attention=config.use_attention,
                                      final_pool_shape=config.final_pool_shape,
                                      kernel_shape=config.conv_kernel_shape,
                                      head_multiplier=head_multiplier,
                                      quadratic_constraint_type=config.get("quadratic_constraint_type", "semi_pos_def"),
                                      debug=config.debug != -1)

        elif config.network_type == "mlp":
            energy_obj = CondMlpEnergy(input_size=config.n_dims,
                                       body_hidden_size=config.mlp_hidden_size,
                                       body_output_size=config.mlp_output_size,
                                       num_blocks=config.mlp_n_blocks,
                                       act_name=config.activation_name,
                                       use_residual=config.use_residual_mlp,
                                       max_spectral_norm_params=config.get("max_spectral_norm_params", None),
                                       reg_coef=config.energy_reg_coef,
                                       dropout_params=config.dropout_params,
                                       bridge_idxs=bridge_idxs,
                                       max_num_ratios=max_num_ratios,
                                       use_cond_scale_shift=config.use_cond_scale_shift,
                                       head_type=config.head_type,
                                       use_single_head=config.get("use_single_head", False),
                                       head_multiplier=head_multiplier,
                                       quadratic_constraint_type=config.get("quadratic_constraint_type", "semi_pos_def")
                                       )
        else:
            raise ValueError("Must specify 'network_type' in config file")

    else:
        energy_obj = SeparableEnergy(bridge_idxs=bridge_idxs,
                                     max_num_ratios=max_num_ratios,
                                     config=config,
                                     only_f=eval_only_f
                                     )
    return energy_obj


# noinspection PyUnresolvedReferences
def plot_per_ratio_and_datapoint_diagnostics(sess,
                                             metric_op,
                                             num_ratios,
                                             datasets,
                                             data_splits,
                                             save_dir,
                                             dp,
                                             config,
                                             data_pholder=None,
                                             feed_dict=None,
                                             name="neg_e",
                                             feed_dict_fn=None
                                             ):
    if feed_dict is None: feed_dict = {}
    for data, split in zip(datasets, data_splits):

        diag_save_dir = os.path.join(save_dir, "{}_diagnostics/{}/".format(name, split))
        os.makedirs(diag_save_dir, exist_ok=True)

        op_per_ratio_and_datapoint = tf_batched_operation(sess=sess,
                                                          ops=metric_op,
                                                          n_samples=len(data),
                                                          batch_size=config.n_batch // num_ratios,
                                                          data_pholder=data_pholder,
                                                          data=data,
                                                          const_feed_dict=feed_dict,
                                                          feed_dict_fn=feed_dict_fn
                                                          )

        op_per_ratio_and_datapoint = op_per_ratio_and_datapoint[:len(data)]  # (n_data, n_ratios)

        # for each ratio, get 5_stat_sum and histogram of metric
        for i in range(op_per_ratio_and_datapoint.shape[1]):
            x = op_per_ratio_and_datapoint[:, i]
            five_stat_and_hist(x, str(i), diag_save_dir)

        op_per_datapoint = np.sum(op_per_ratio_and_datapoint, axis=1)  # (n, )
        five_stat_and_hist(op_per_datapoint, "all", diag_save_dir)

        if "neg_e" in name:
            # plot the top ~250 highest and lowest ranked imgs
            sort_idx = np.argsort(op_per_datapoint)
            sorted_data = data[sort_idx]
            use_pca = "pca" in config.data_args and config.data_args["pca"]
            if (config.dataset_name in IMG_DATASETS) or use_pca:

                sorted_data = revert_data_preprocessing(sorted_data, dp, is_wmark_input=False)

                disp_imdata(sorted_data, config.dataset_name, num_pages=5,
                            dir_name=os.path.join(diag_save_dir, "low_neg_energy_img_data/"))
                disp_imdata(sorted_data[::-1], config.dataset_name, num_pages=5,
                            dir_name=os.path.join(diag_save_dir, "high_neg_energy_img_data/"))

    val_op_per_ratio_and_datapoint = op_per_ratio_and_datapoint

    return val_op_per_ratio_and_datapoint
# noinspection PyUnresolvedReferences


def load_flow(sess, config, flow_id):

    flow_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='flow/')
    saver = tf.train.Saver(var_list=flow_vars, max_to_keep=2, save_relative_paths=True)

    ckpt_dir = os.path.join(project_root, "saved_models/{}/flow/{}/model/".format(config.dataset_name, flow_id))
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))


# noinspection PyUnresolvedReferences
def load_model(sess, epoch_idx, config, flow_mode=None):
    logger = logging.getLogger("tf")
    logger.info("Restoring model!")

    energy_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='tre_model/')
    saver = tf.train.Saver(var_list=energy_vars, max_to_keep=2, save_relative_paths=True)

    if epoch_idx == "best":  # load best model found from early stopping
        load_path = os.path.join(config.save_dir, "model/")
        saver.restore(sess, tf.train.latest_checkpoint(load_path))

    else:  # load a model from a specific epoch
        load_path = os.path.join(config.save_dir,
                                 "model/every_x_epochs/{}.ckpt".format(epoch_idx))
        saver.restore(sess, load_path)


def load_flow_vars (sess, graph_flow_id, folder_flow_id, config, flow_dir_name):
    flow_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope='tre_model/flow_{}'.format(graph_flow_id))

    saver = tf.train.Saver(var_list=flow_vars, max_to_keep=2, save_relative_paths=True)
    load_path = os.path.join(project_root, "saved_models", config.dataset_name, flow_dir_name,
                             "flow{}".format(folder_flow_id), "model/")

    saver.restore(sess, tf.train.latest_checkpoint(load_path))
