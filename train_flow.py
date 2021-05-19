from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

from __init__ import project_root, density_data_root
from experiment_ops import plot_chains, build_flow
from mcmc.mcmc_utils import build_mcmc_chain
from scipy.stats import norm, iqr
from utils.misc_utils import *
from utils.tf_utils import *
from utils.experiment_utils import *
from utils.plot_utils import *


# noinspection PyUnresolvedReferences
def build_placeholders(conf):
    data_args = conf.data_args
    if (data_args is not None) and ("img_shape" in data_args) and (data_args["img_shape"] is not None):
        default_data = tf.constant(1.0, dtype=tf.float32, shape=(1, *data_args["img_shape"]))
        data = tf.placeholder_with_default(default_data, (None, *data_args["img_shape"]), "data")
        mcmc_init = tf.placeholder(dtype=tf.float32, shape=(None, *data_args["img_shape"]), name="mcmc_init")
        mode_init = tf.placeholder(dtype=tf.float32, shape=(10, *data_args["img_shape"]), name="mode_init")
    else:
        default_data = tf.constant(1.0, dtype=tf.float32, shape=(1, conf.n_dims))
        data = tf.placeholder_with_default(default_data, (None, conf.n_dims), "data")
        mcmc_init = tf.placeholder(dtype=tf.float32, shape=(None, conf.n_dims), name="mcmc_init")
        mode_init = tf.placeholder(dtype=tf.float32, shape=(10, conf.n_dims), name="mode_init")

    lr = tf.placeholder_with_default(5e-4, (), name="learning_rate")
    keep_prob = tf.placeholder_with_default(1.0, (), name="dropout_keep_prob")
    is_training_bool = tf.placeholder_with_default(False, shape=(), name="is_training_bool")
    n_samples = tf.placeholder_with_default(1, (), name="n_samples")

    return data, lr, is_training_bool, keep_prob, n_samples, mcmc_init, mode_init


# noinspection PyUnresolvedReferences
def build_mle_loss(log_prob, lr, config):
    """Estimate flow params with maximum likelihood estimation"""
    l2_loss = tf.losses.get_regularization_loss(scope="flow")
    nll = -tf.reduce_mean(log_prob)
    reg_nll = nll + l2_loss

    flow_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="flow")

    # if config.flow_type == "GaussianCopula":
    #     flow_params = [p for p in flow_params if
    #                    ("gauss_copula_cholesky" not in p.name) and
    #                    ("gauss_copula_mean" not in p.name)]

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="flow")
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(lr)
        optim_op = optimizer.minimize(reg_nll, var_list=flow_params)

    return nll, optim_op


def build_flow_graph(config):
    data, lr, is_training_bool, keep_prob, n_samples, mcmc_init, mode_init = build_placeholders(config)

    flow, flow_log_p = build_flow(config, data, flow_training_bool=is_training_bool,
                                  flow_keep_prob=keep_prob, flow_reg_coef=config.flow_reg_coef)
    noise_samples = flow.sample(n_samples)

    # learn the parameters of the flow with maximum likelihood estimation
    flow_nll, flow_optim_op = build_mle_loss(flow_log_p, lr, config)
    noise_log_prob_own_samples = flow.log_prob(noise_samples)

    inverted_data = flow.inverse(data)
    reconstructed_data = flow.forward(inverted_data)

    metric_nll, update_nll = tf.metrics.mean(flow_nll)

    if config.run_mcmc_sampling:
        z_space_init = flow.inverse(mcmc_init)
        mcmc_results = build_mcmc_chain(
            target_log_prob_fn=flow.base_dist.log_prob,
            initial_states=z_space_init,
            n_samples_to_keep=config.n_mcmc_samples_to_keep,
            thinning_factor=0,
            mcmc_method="nuts",
            # mcmc_method="hmc",
            step_size=0.02,
            n_adaptation_steps=int(config.n_mcmc_samples_to_keep/2)
        )
        mcmc_results[0] = flow.forward(mcmc_results[0])

    if config.run_mode_finder:
        if (data_args is not None) and ("img_shape" in data_args) and (data_args["img_shape"] is not None):
            event_shp = data_args["img_shape"]
        else:
            event_shp = [config.n_dims]

        mode_vars = tf.get_variable('input_images', shape=[10, *event_shp], dtype=tf.float32, trainable=True)
        mode_init_assign = mode_vars.assign(mode_init)

        neg_log_prob_mode_vars = -flow.log_prob(mode_vars)
        av_neg_log_prob_mode_vars = tf.reduce_mean(neg_log_prob_mode_vars)

        optimizer = tf.train.AdamOptimizer(lr)
        mode_finder_optim_op = optimizer.minimize(av_neg_log_prob_mode_vars, var_list=[mode_vars])

    return AttrDict(locals())


def train(g, sess, train_dp, val_dp, saver, config):
    logger = logging.getLogger("tf")

    model_dir = config.save_dir + "model/"
    os.makedirs(model_dir, exist_ok=True)

    start_epoch_idx = config.get("epoch_idx", -1)
    config["epoch_idx"] = start_epoch_idx

    if config.flow_type == 'GLOW' and start_epoch_idx == -1:
        logger.info("initialising glow...")  # This is required for stability of glow
        init_batch_size = g.flow.flow.hparams.init_batch_size
        sess.run(g.flow.glow_init, feed_dict={g.data: train_dp.data[:init_batch_size]})

    config["n_epochs_until_stop"], config["best_val_loss"] = config.patience, np.inf
    for _ in range(start_epoch_idx, config.n_epochs):

        lr = pre_epoch_events(config, logger)

        for j, batch in enumerate(train_dp):
            feed_dict = {g.data: batch, g.keep_prob: config.flow_keep_prob, g.is_training_bool: True, g.lr: lr}
            _ = sess.run(g.flow_optim_op, feed_dict=feed_dict)

        config.n_epochs_until_stop -= 1

        # Evaluate model
        eval_model(g, sess, train_dp, val_dp, config, all_train_data=False)

        # Check early stopping criterion
        save_path = model_dir + "{}.ckpt".format(config.epoch_idx)
        stop, _ = check_early_stopping(saver, sess, save_path, config)

        if stop:
            break  # early stopping triggered

    logger.info("Finished training model!")
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    # if config.flow_type == "GaussianCopula":
    #     fit_mvn_for_gauss_copula(sess, train_dp, config, logger, use_rank_approach=True)
    #     save_path = model_dir + "{}.ckpt".format(config.n_epochs)
    #     saver.save(sess, save_path)

    eval_model(g, sess, train_dp, val_dp, config, all_train_data=True)


def pre_epoch_events(config, logger):

    config["epoch_idx"] += 1
    save_config(config)

    lr = 0.5 * config.flow_lr * (1 + np.cos((config.epoch_idx / config.n_epochs) * np.pi))
    logger.info("LEARNING RATE IS NOW {}".format(lr))

    return lr


# def fit_mvn_for_gauss_copula(sess, train_dp, config, logger, use_rank_approach=False):
#     logger.info("Fitting Gauss Copula covariance matrix.")
#     data = train_dp.data.reshape(-1, config.n_dims)
#     if use_rank_approach:
#         z_data = np.zeros_like(data)  # (n, d)
#         for j in range(config.n_dims):
#             xi = data[:, j]
#             order = np.argsort(xi)
#             ranks = np.argsort(order)
#             xi_ranks = (ranks + 1) / (len(xi) + 1)
#             z_data[:, j] = norm.ppf(xi_ranks)
#         cov = (1/len(z_data)) * np.dot(z_data.T, z_data)  # (d, d)
#
#     else:
#         cov = np.cov(data, rowvar=False, bias=True)
#
#     cholesky = np.linalg.cholesky(cov)
#     idxs = np.diag_indices_from(cholesky)
#     cholesky[idxs] = np.log(cholesky[idxs])
#
#     flow_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="flow")
#     chol_var = [p for p in flow_params if ("gauss_copula_cholesky" in p.name)][0]
#     sess.run(tf.assign(chol_var, cholesky))


# noinspection PyUnresolvedReferences,PyTypeChecker
def eval_model(g, sess, train_dp, val_dp, config, all_train_data=False):
    """Each epoch, evaluate the MLE loss on train+val set"""

    # only use a subset of train data (of same size as val data) to evaluate model
    if not all_train_data: train_dp.max_num_batches = val_dp.num_batches

    trn_loss = eval_metrics(g, sess, train_dp)
    val_loss = eval_metrics(g, sess, val_dp)

    # reset number of batches in training data providers
    train_dp.max_num_batches = -1

    trn_bpd = convert_to_bits_per_dim(-trn_loss + np.mean(train_dp.ldj), config.n_dims, val_dp.source.original_scale)
    val_bpd = convert_to_bits_per_dim(-val_loss + np.mean(val_dp.ldj), config.n_dims, val_dp.source.original_scale)

    logger = logging.getLogger("tf")
    logger.info("Epoch {}".format(config.get("epoch_idx", -1)))
    logger.info("trn NLL {:0.3f} / BPD {:0.3f} |  "
                "val NLL {:0.3f} / BPD {:0.3f}".format(trn_loss, trn_bpd, val_loss, val_bpd))

    config["current_val_loss"] = val_loss

    if "2d" in config.dataset_name or "1d" in config.dataset_name:
        gridsize = "large"
        tst_grid_coords = getattr(val_dp.source_1d_or_2d, "tst_coords_{}".format(gridsize))
        logprobs = sess.run(g.flow_log_p, {g.data: tst_grid_coords, g.keep_prob: 1.0})  # (n_tst, n_ratios)
        logprobs = np.expand_dims(logprobs, axis=1)
        val_dp.source_1d_or_2d.plot_logratios(logprobs, config.save_dir + "figs/",
                                           "{}_density_plots".format(gridsize), gridsize=gridsize)
        val_dp.source_1d_or_2d.plot_logratios(logprobs, config.save_dir + "figs/",
                                           "{}_log_density_plots".format(gridsize), log_domain=True, gridsize=gridsize)


def eval_metrics(g, sess, dp):
    sess.run(tf.local_variables_initializer())
    for batch in dp:
        feed_dict = {g.data: batch, g.keep_prob: 1.0}
        sess.run(g.update_nll, feed_dict=feed_dict)
    loss = sess.run(g.metric_nll)
    return loss


# noinspection PyUnresolvedReferences
def sample_and_assess_diagnostics(g, sess, dp, config):

    logger = logging.getLogger("tf")
    fig_dir = config.save_dir + "figs/"
    os.makedirs(fig_dir, exist_ok=True)

    sample_log_probs, samples = sample_from_model(sess, g, config, logger)

    plot_density_hists_samples_vs_data(config, dp, fig_dir, g, sample_log_probs, sess)

    save_and_visualise_samples(g, sess, samples, dp, config, fig_dir)

    if config.run_mcmc_sampling:
        logger.info("running MCMC sampler...")
        run_mcmc_sampler(samples[:100], config, dp, fig_dir, g, logger, sess)

    if config.run_mode_finder:
        logger.info("finding mode(s) of distribution via gradient ascent")
        find_modes(config, dp, fig_dir, g, sess, logger)


def sample_from_model(sess, g, config, logger):
    total_num_samples = config.num_samples
    res = tf_batched_operation(sess=sess,
                               ops=[g.noise_samples, g.noise_log_prob_own_samples],
                               n_samples=total_num_samples,
                               batch_size=config.n_batch,
                               const_feed_dict={g.n_samples: config.n_batch})
    samples, sample_log_probs = res
    logger.info("min and max of samples: {}, {}".format(samples.min(), samples.max()))
    logger.info("av log prob of samples: {}".format(sample_log_probs.mean()))
    return sample_log_probs, samples


def plot_density_hists_samples_vs_data(config, dp, fig_dir, g, sample_log_probs, sess):
    data_logp = tf_batched_operation(sess=sess,
                                     ops=g.flow_log_p,
                                     n_samples=dp.data.shape[0],
                                     batch_size=config.n_batch,
                                     data_pholder=g.data,
                                     data=dp.data)

    # compute lower/upper quartile -/+ IQR of density of samples
    l_quartile = np.percentile(sample_log_probs, 25)
    u_quartile = np.percentile(sample_log_probs, 75)
    i_range = u_quartile - l_quartile
    tukey_range = [l_quartile - (1.5*i_range), u_quartile + (1.5*i_range)]

    fig, ax = plt.subplots(1, 1)
    h1 = plot_hist(sample_log_probs, alpha=0.5, ax=ax, color='r', label='samples')
    h2 = plot_hist(data_logp, alpha=0.5, ax=ax, color='b', label='data')
    max_val = max(h1.max(), h2.max())
    ax.plot(np.ones(128)*tukey_range[0], np.linspace(0, max_val, 128), linestyle='--', c='r', label="quartile +/- 1.5*IQR")
    ax.plot(np.ones(128)*tukey_range[1], np.linspace(0, max_val, 128), linestyle='--', c='r')
    ax.legend()
    save_fig(fig_dir, "density_of_data_vs_samples")


def run_mcmc_sampler(init_states, config, dp, fig_dir, graph, logger, sess):

    results = sess.run(graph.mcmc_results, feed_dict={graph.mcmc_init: init_states})
    all_chains, accept_rate, final_ss, nuts_leapfrogs = results

    logger.info("MCMC sampling:")
    logger.info("Final acceptance rate: {}".format(accept_rate))
    logger.info("Final step size: {}".format(final_ss[0]))
    if nuts_leapfrogs: logger.info("Num nuts leapfrogs: {}".format(nuts_leapfrogs))

    logger.info("saving chains to disk...")
    np.savez_compressed(fig_dir + "mcmc_chains", samples=all_chains)

    # create various plots to analyse the chains
    logger.info("plotting chains...")
    plot_chains(all_chains,
                "mcmc_samples",
                fig_dir,
                dp=dp,
                config=config,
                graph=graph,
                sess=sess,
                rank_op=graph.flow_log_p,
                plot_hists=True)


def find_modes(config, dp, fig_dir, g, sess, logger):

    mode_dir = os.path.join(fig_dir, "modes/")
    sess.run(g.mode_init_assign, feed_dict={g.mode_init: dp.data[:10]})
    for i in range(config.num_mode_finding_iters):

        if i % 10000 == 0:
            cur_modes, av_nll  = sess.run([g.mode_vars, g.av_neg_log_prob_mode_vars],
                                          feed_dict={g.lr: config.mode_finding_lr})

            logger.info("mode finding iter {}: nll is {}".format(i, av_nll))
            plot_chains_main(np.expand_dims(cur_modes, axis=1),
                             name="iter_{}".format(i),
                             save_dir=mode_dir,
                             dp=dp,
                             config=config)

        sess.run(g.mode_finder_optim_op)


def save_and_visualise_samples(g, sess, model_samples, dp, config, fig_dir):

    if config.plot_sample_histograms:
        n_samples = len(model_samples)
        data_to_plot = [model_samples.reshape(n_samples, -1),  dp.data[:n_samples].reshape(n_samples, -1)]
        labels, colours = ["model", "data"], ["red", "blue"]
        plot_hists_for_each_dim(n_dims_to_plot=config.n_dims,
                                data=data_to_plot,
                                labels=labels,
                                colours=colours,
                                dir_name=fig_dir + "hists_and_scatters/",
                                filename="data_vs_flow",
                                increment=10,
                                include_scatter=True
                                )
        plot_hists_for_each_dim(n_dims_to_plot=config.n_dims,
                                data=data_to_plot,
                                labels=labels,
                                colours=colours,
                                dir_name=fig_dir + "hists/",
                                filename="data_vs_flow",
                                increment=49,
                                include_scatter=False
                                )

    num_imgs_plot = min(100, len(model_samples))
    plotting_samples = model_samples[:num_imgs_plot]
    sample_shp = plotting_samples.shape

    plotting_samples = plotting_samples.reshape(num_imgs_plot, 1, *sample_shp[1:])  # insert 1 to match plot_chains api
    plotting_data = dp.data[:num_imgs_plot].reshape(num_imgs_plot, 1, *sample_shp[1:])

    plot_chains(plotting_samples, "flow_samples", fig_dir, dp, config, g, sess, plot_hists=False)
    plot_chains(plotting_data, "data_samples", fig_dir, dp, config, g, sess, plot_hists=False)


def save_trn_or_val(dir_root, filename, model_samples, model_samples_log_prob, which_set="train/"):
    save_dir = dir_root + which_set
    os.makedirs(save_dir, exist_ok=True)
    file_path = save_dir + filename
    np.savez_compressed(file_path, data=model_samples, log_probs=model_samples_log_prob)


def save_trimmed_datasets(config, graph, sess, dp, which_set):

    ordered_data, sort_idxs = plot_chains(chains=np.expand_dims(dp.data, axis=1),
                                          name="density_ordered_data",
                                          save_dir=config.save_dir + "figs/",
                                          dp=dp,
                                          config=config,
                                          graph=graph,
                                          sess=sess,
                                          rank_op=graph.flow_log_p,
                                          ret_chains=True)

    logger = logging.getLogger("tf")
    ordered_data = np.squeeze(ordered_data)
    logger.info("N datapoints: {}".format(len(ordered_data)))

    data_dir = path_join(density_data_root, config.dataset_name, which_set)
    os.makedirs(data_dir, exist_ok=True)

    np.savez(path_join(data_dir, "{}_sort_idxs".format(config.flow_type)), sort_idxs=sort_idxs)


def print_out_loglik_results(all_dict, logger):
    for dic in all_dict:
        for key, val in dic.items():
            logger.info("----------------------------")
            logger.info(key)
            logger.info("mean / median / std / min / max")
            logger.info(five_stat_sum(val))
            logger.info("----------------------------")


def make_config():
    parser = ArgumentParser(description='Uniformize marginals of a dataset', formatter_class=ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--config_path', type=str, default="1d_gauss/flow/0")
    # parser.add_argument('--config_path', type=str, default="2d_spiral/flow/0")
    # parser.add_argument('--config_path', type=str, default="mnist/flow/0")
    # parser.add_argument('--config_path', type=str, default="mnist/flow/20200406-1408_0/config")

    parser.add_argument('--restore_model', type=str, default=-1)
    parser.add_argument('--only_sample', type=int, default=-1)  # -1 means false, otherwise true

    parser.add_argument('--num_samples', type=int, default=150)  # -1 means false, otherwise true
    parser.add_argument('--run_mcmc_sampling', type=int, default=-1)  # -1 means false, otherwise true
    parser.add_argument('--n_mcmc_samples_to_keep', type=int, default=10)  # -1 means false, otherwise true
    parser.add_argument('--run_mode_finder', type=int, default=-1)  # -1 means false, otherwise true
    parser.add_argument('--num_mode_finding_iters', type=int, default=100000)  # -1 means false, otherwise true
    parser.add_argument('--mode_finding_lr', type=int, default=10000)  # -1 means false, otherwise true
    parser.add_argument('--plot_sample_histograms', type=int, default=-1)  # -1 means false, otherwise true

    parser.add_argument('--flow_reg_coef', type=float, default=1e-6)
    parser.add_argument('--glow_temperature', type=float, default=1.0)
    parser.add_argument('--frac', type=float, default=1.0)
    parser.add_argument('--debug', type=int, default=-1)
    args = parser.parse_args()

    root = "saved_models" if args.restore_model != -1 else "configs"
    with open(project_root + "{}/{}.json".format(root, args.config_path)) as f:
        config = json.load(f)

    if args.restore_model == -1:
        config = merge_dicts(*list(config.values()))  # json is 2-layers deep, flatten it
    rename_save_dir(config)
    config.update(vars(args))

    config["restore_model"] = True if config["restore_model"] != -1 else False
    config["only_sample"] = True if config["only_sample"] != -1 else False
    config["run_mcmc_sampling"] = True if config["run_mcmc_sampling"] != -1 else False
    config["run_mode_finder"] = True if config["run_mode_finder"] != -1 else False
    config["plot_sample_histograms"] = True if config["plot_sample_histograms"] != -1 else False

    if config["flow_type"] == "GLOW":
        assert config["data_args"]["img_shape"] is not None, "must specify img shape to use GLOW"

    if config["only_sample"]:
        assert config["restore_model"], "Must specify restore_model if only_sample==True!"

    if args.debug != -1:
        config["n_epochs"] = 1
        config["frac"] = 0.01
        # config["flow_hidden_size"] = 64
        # config["glow_depth"] = 2

    if "flow/" not in config["save_dir"]:
        s = config["save_dir"].split("/")
        s.insert(-2, "flow")
        config["save_dir"] = '/'.join(s)

    save_config(config)
    globals().update(config)

    return AttrDict(config)


# noinspection PyUnresolvedReferences,PyTypeChecker
def main():
    """Train a flow-based neural density estimator with maximum likelihood estimation"""
    make_logger()
    logger = logging.getLogger("tf")
    np.set_printoptions(precision=3)
    tf.reset_default_graph()

    # load a config file whose contents are added to globals(), making them easily accessible elsewhere
    config = make_config()

    train_dp, val_dp = load_data_providers_and_update_conf(config)

    # create a dictionary whose keys are tensorflow operations that can be accessed like attributes e.g graph.operation
    graph = build_flow_graph(config)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        flow_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='flow')
        saver = tf.train.Saver(var_list=flow_vars, max_to_keep=2, save_relative_paths=True)

        if config.restore_model:
            rel_path = "saved_models/{}/model/".format("/".join(config["config_path"].split("/")[:-1]))
            saver.restore(sess, tf.train.latest_checkpoint(project_root + rel_path))
            logger.info("Model restored!")
            eval_model(graph, sess, train_dp, val_dp, config, all_train_data=True)


        if not config.only_sample:
            train(graph, sess, train_dp, val_dp, saver, config)

        sample_and_assess_diagnostics(graph, sess, train_dp, config)


    save_config(config)
    logger.info("Finished!")


if __name__ == "__main__":
    main()
