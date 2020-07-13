import numpy as np
import tensorflow as tf

from copy import deepcopy
from utils.misc_utils import *
from utils.tf_utils import *
from utils.experiment_utils import *
from utils.plot_utils import *
from waymark_ops import get_waymark_and_bridge_idxs_for_epoch_i


def analyse_objective_for_1d_gauss_multiple_sample_sizes(config, graph, sess, train_dp, get_feed_dict):
    theta_diffs = []
    original_data = deepcopy(train_dp.data)
    for sample_size in [10, 100, 1000, 10000, 100000]:
        train_dp.data = original_data[:sample_size]
        analyse_objective_fn_for_1d_gauss(graph, sess, train_dp, config, get_feed_dict, theta_diffs=theta_diffs)

    print("all theta diffs: ", theta_diffs)

    fig_dir = path_join(config.save_dir, "figs/")
    os.makedirs(fig_dir, exist_ok=True)

    if len(config.initial_waymark_indices) == 2:
        filename = "one_ratio_theta_diffs"
    else:
        filename = "tre_theta_diffs"
    np.savez(path_join(fig_dir, filename), theta_diffs=np.array(theta_diffs))


def analyse_objective_fn_for_1d_gauss(g, sess, train_dp, config, get_feed_dict, theta_diffs=None, do_plot=False, do_load=None):

    energy_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="tre_model")

    graph_scale_param = [v for v in energy_params if "b_all" in v.name][0]
    graph_quadratic_coef = [v for v in energy_params if "Q_all" in v.name][0]

    waymark_idxs, bridge_idxs = get_waymark_and_bridge_idxs_for_epoch_i(config, -1)
    wmark_coeffs = sess.run(g.waymark_construction_results.noise_multipliers,
                            feed_dict={g.waymark_idxs: waymark_idxs, g.bridge_idxs: bridge_idxs}
                            )
    num_ratios = shape_list(graph_scale_param)[0]

    true_scales = np.zeros(num_ratios)
    true_quadratic_coefs = np.zeros(num_ratios)
    estimated_quadratic_coefs = np.zeros(num_ratios)
    for i in range(num_ratios):

        if do_load:
            filename = "one_ratio_1d_analysis_arrays.npz" if num_ratios == 1 else "tre_ratio_{}_1d_analysis_arrays.npz".format(i)
            array_path = path_join(config.save_dir, filename)
            loss_vals, sigma1, sigma2, true_quadratic_coef, x_axis = _load_obj_fn_arrays(array_path)
        else:
            loss_vals, sigma1, sigma2, true_quadratic_coef, x_axis = \
                compute_obj_fn_for_one_ratio(g,
                                             sess,
                                             i,
                                             graph_quadratic_coef,
                                             graph_scale_param,
                                             true_quadratic_coefs,
                                             estimated_quadratic_coefs,
                                             wmark_coeffs,
                                             true_scales,
                                             num_ratios,
                                             train_dp,
                                             config,
                                             get_feed_dict)

        if do_plot:
            plot_densities_and_obj_fn_for_1d_gauss(i, x_axis, loss_vals, true_quadratic_coef, num_ratios, sigma1, sigma2, config.loss_function)

    theta_est = 0.5 * logsumexp(2 * estimated_quadratic_coefs)
    theta_true = 0.5 * logsumexp(2 * true_quadratic_coefs)

    print("Estimated quadratic coefficient of TRE model is : {}".format(theta_est))
    print("True quadratic coefficient is : {}".format(theta_true))
    print("|theta^* - theta_est| = ", np.abs(theta_true - theta_est))
    if theta_diffs is not None:
        theta_diffs.append(np.abs(theta_true - theta_est))


def _load_obj_fn_arrays(load_path):
    loaded = np.load(load_path)
    loss_vals = loaded["loss_vals"]
    sigma1 = loaded["sigma1"]
    sigma2 = loaded["sigma2"]
    true_quadratic_coef = loaded["true_quadratic_coef"]
    x_axis = loaded["x_axis"]
    return loss_vals, sigma1, sigma2, true_quadratic_coef, x_axis


def compute_obj_fn_for_one_ratio(g,
                                 sess,
                                 i,
                                 graph_quadratic_coef,
                                 graph_scale_param,
                                 true_quadratic_coefs,
                                 estimated_quadratic_coefs,
                                 wmark_coeffs,
                                 true_scales,
                                 num_ratios,
                                 train_dp,
                                 config,
                                 get_feed_dict):

    data_sigma = config.data_args["std"]
    noise_sigma = config.noise_dist_gaussian_stds[0]

    alpha_i, beta_i = wmark_coeffs[i], wmark_coeffs[i + 1]
    sigma1 = (((1 - alpha_i ** 2) * data_sigma ** 2) + (alpha_i ** 2 * noise_sigma ** 2)) ** 0.5
    sigma2 = (((1 - beta_i ** 2) * data_sigma ** 2) + (beta_i ** 2 * noise_sigma ** 2)) ** 0.5

    true_scales[i] = np.log(sigma1) - np.log(sigma2)
    sess.run(tf.assign(graph_scale_param, true_scales))

    r = (sigma1 ** 2) / (sigma2 ** 2)
    true_quadratic_coef = 0.5 * (-np.log(2) - 2 * np.log(sigma1) + np.log(1 - r))

    def _feed_dict_fn(j, n, b):
        batch = train_dp.data[j:min(j + b, n), ...]
        return get_feed_dict(g, sess, train_dp, batch, config, train=False)

    loss_vals = []

    # for results in the paper, I used a grid with 500 points. However, this takes a while, so I'll
    # leave a default value of 150 to enable faster approximate reproduction of results.
    grid_size = 150
    print("Using grid of {} points to evaluate objective(s)".format(grid_size))

    x_axis = [i for i in np.linspace(0.3 * true_quadratic_coef, 1.5 * true_quadratic_coef, grid_size)]
    x_axis.append(true_quadratic_coef)
    x_axis.sort()
    for log_std in x_axis:
        true_quadratic_coefs[i] = log_std + 5.0  # add 5 to undo a -5 operation used in the tensorflow graph
        sess.run(tf.assign(graph_quadratic_coef, true_quadratic_coefs.reshape(-1, 1, 1)))
        loss = tf_batched_operation(sess,
                                    tf.reshape(g.val_loss, [1, -1]),
                                    len(train_dp.data),
                                    config.n_batch,
                                    feed_dict_fn=_feed_dict_fn
                                    )
        av_loss = loss.mean(axis=0)  # (n_ratios, )
        loss_vals.append(av_loss[i])

    x_axis = np.array(x_axis)
    min_x = x_axis[np.argmin(np.array(loss_vals))]
    estimated_quadratic_coefs[i] = min_x
    true_quadratic_coefs[i] = true_quadratic_coef

    save_dir = path_join(config.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    filename = "one_ratio_1d_analysis_arrays" if num_ratios == 1 else "tre_ratio_{}_1d_analysis_arrays".format(i)
    np.savez(path_join(save_dir, filename),
             loss_vals=np.array(loss_vals),
             sigma1=np.array(sigma1),
             sigma2=np.array(sigma2),
             true_quadratic_coef=np.array(true_quadratic_coef),
             x_axis=x_axis
             )

    return loss_vals, sigma1, sigma2, true_quadratic_coef, x_axis


def plot_densities_and_obj_fn_for_1d_gauss(i, x_axis_for_loss_fn, loss_vals, true_quadratic_coef, num_ratios, sigma1, sigma2, loss_type):

    if num_ratios != 1:
        set_all_fontsizes(HUGE_SIZE)

    figsize = (13, 3) if num_ratios == 1 else (4.5, 6.5)
    layout = [1, 2] if num_ratios == 1 else [2, 1]
    fig, axs = plt.subplots(*layout, figsize=figsize)
    axs = axs.ravel()
    colour = 'b' if num_ratios == 1 else 'r'

    # PLOT DENSITIES / RATIO
    ax = axs[0]
    x_axis = [i for i in np.linspace(-3, 3, 512)]
    x_axis.append(0.)
    x_axis += [i for i in np.linspace(-1e-2, 1e-2, 512)]
    x_axis = np.array(sorted(x_axis))

    p1 = norm(loc=0.0, scale=sigma1).pdf(x_axis)
    p2 = norm(loc=0.0, scale=sigma2).pdf(x_axis)
    label = r"$p(x)$" if i==0 else r"$p_{%s}(x)$" % i
    ax.plot(x_axis, p1, label=label, c='dimgrey')

    label = r"$q(x)$" if i == num_ratios-1 else r"$p_{%s}(x)$" % (i + 1)
    ax.plot(x_axis, p2, label=label, c='darkgrey')

    ratio_vals = (sigma2 / sigma1) * np.exp(((x_axis ** 2) / 2) * ((1 / (sigma2 ** 2)) - (1 / (sigma1 ** 2))))
    if num_ratios == 1:
        label = r"$\frac{p(x)}{q(x)}$"
    else:
        if i == 0:
            label = r"$\frac{p(x)}{p_1(x)}$"
        elif i == num_ratios - 1:
            label = r"$\frac{p_{%s}(x)}{q(x)}$" % i
        else:
            label = r"$\frac{p_{%s}(x)}{p_{%s}(x)}$" % (i, i + 1)
    ax.plot(x_axis, ratio_vals, label=label, c='b' if num_ratios == 1 else 'r')

    ax.set_yscale('symlog', linthreshy=0.01)
    ax.set_xscale('symlog', linthreshx=1e-2)

    # only keep first & last ticks
    if num_ratios == 1:
        ax.set_yticks(ax.get_yticks()[::2])
    else:
        ax.set_xticks([ax.get_xticks()[0], ax.get_xticks()[-1]])
        ax.set_yticks([ax.get_yticks()[0], ax.get_yticks()[-1]])

    ax.set_xlabel(r"$x$")
    if i== 0: ax.set_ylabel("density/ratio value")
    # ax.set_title("Single ratio estimation" if num_ratios == 1)
    ax.legend(loc='upper left')


    # PLOT OBJECTIVE FUNCTION
    ax = axs[1]

    loss_vals = np.array(loss_vals)
    idxs = loss_vals < 1.5
    loss_vals = loss_vals[idxs]

    x_axis_for_loss_fn = np.array(x_axis_for_loss_fn)
    x_axis_for_loss_fn = x_axis_for_loss_fn[idxs]
    min_x = x_axis_for_loss_fn[np.argmin(loss_vals)]

    label = r"$\mathcal{L}^n(\theta)$" if num_ratios == 1 else r"$\mathcal{L}^n_{%s}(\theta_{%s})$" % (i, i)
    ax.plot(x_axis_for_loss_fn, loss_vals, label=label, c=colour)

    label = r"$\hat{\theta}$" if num_ratios == 1 else r"$\hat{\theta}_{%s}$" % i
    ax.plot(np.ones(128) * min_x, np.linspace(min(loss_vals), max(loss_vals), 128), label=label, linestyle="--", c=colour)

    label = r"$\theta^*$" if num_ratios == 1 else r"$\theta_{%s}^*$" % i
    ax.plot(np.ones(128) * true_quadratic_coef, np.linspace(min(loss_vals), max(loss_vals), 128), label=label, linestyle="--", c='k')

    if num_ratios == 1:
        label = r"$\theta_{TRE}$"  # these values were obtained by running TRE, and subsequently hardcoding the results here
        if loss_type == "logistic":
            tre_est = 13.51
        elif loss_type == "nwj":
            tre_est = 13.56
        elif loss_type == "lsq":
            tre_est = 13.40
        else:
            raise ValueError
        ax.plot(np.ones(128) * tre_est, np.linspace(min(loss_vals), max(loss_vals), 128), label=label, linestyle="--", c='r')

    # only keep first, middle & final y-ticks
    if num_ratios != 1:
        if i == 0:
            ax.set_yticks([ax.get_yticks()[1], ax.get_yticks()[-2]])
        else:
            ax.set_yticks([])

    ax.set_xlabel(r"$\theta$" if num_ratios == 1 else r"$\theta_{%s}$" % i)
    if i == 0: ax.set_ylabel("logistic loss")
    ax.legend(loc='upper right')

    fig.tight_layout()
    fig_dir = os.path.join(project_root, "saved_models/1d_gauss/results/{}/".format(loss_type))
    filename = "1d_gauss_one_ratio" if num_ratios == 1 else "1d_gauss_tre_ratio_{}".format(i)
    save_fig(fig_dir, filename)
