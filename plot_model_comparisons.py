from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from time import gmtime, strftime
from utils.misc_utils import *
from utils.experiment_utils import *
from utils.plot_utils import *


def plot_model_comparison(metrics_dict, x_var_name, y_var_names, curve_idxs, curve_labels,
                          save_dir, figname, xlabel=None, ylabels=None, curve_colors=None, markers=None):
    for i, y_var_name in enumerate(y_var_names):

        x_var = metrics_dict[x_var_name]
        if y_var_name in metrics_dict:
            y_var = metrics_dict[y_var_name]
        else:
            continue

        fig, ax = plt.subplots(1, 1)
        for j in range(len(curve_idxs)):
            c = curve_colors[j] if curve_colors is not None else None
            m = markers[j] if markers is not None else None

            # if y_var_name == "prenormalised_kl" and "ais_kl" in metrics_dict:
            #     plot_sorted_x_vs_y(ax, x_var, metrics_dict["ais_kl"], curve_labels[j],
            #                        subset_idxs=curve_idxs[j], c=c, alpha=0.4)

            plot_sorted_x_vs_y(ax, x_var, y_var, curve_labels[j], subset_idxs=curve_idxs[j], c=c, m=m)


        if y_var_name in ["prenormalised_kl", "dv_bound", "nwj_bound", "ais_kl", "raise_kl"] and \
                "true_mutual_info" in metrics_dict:
            plot_sorted_x_vs_y(ax, x_var, metrics_dict["true_mutual_info"], "ground truth", subset_idxs=curve_idxs[0], c='k', m='D')

        if y_var_name in ["prenormalised_js", "ais_js"] and "true_js" in metrics_dict:
            plot_sorted_x_vs_y(ax, x_var, metrics_dict["true_js"], "true JS", subset_idxs=curve_idxs[0])

        xlabel = xlabel if xlabel else x_var_name
        ax.set_xlabel(xlabel)
        ylabel = ylabels[i] if ylabels is not None else y_var_name
        ax.set_ylabel(ylabel)
        ax.legend()

        remove_repeated_legends(fig)
        save_fig(save_dir, figname + "_{}_vs_{}".format(x_var_name, y_var_name))


def plot_sorted_x_vs_y(ax, x_var, y_var, label, subset_idxs=None, c=None, m=None, alpha=1.0):
    if subset_idxs is None:
        subset_idxs = np.arange(len(x_var))  # use entire array

    x_var_subset = x_var[subset_idxs]
    sorted_idxs = np.argsort(x_var_subset)

    sorted_x_var_subset = x_var_subset[sorted_idxs]
    sorted_y_var_subset = y_var[subset_idxs][sorted_idxs]

    num_unique = len(np.unique(sorted_x_var_subset))
    if num_unique != len(sorted_x_var_subset):
        plot_x_vs_y_with_errorbars(ax, sorted_x_var_subset, sorted_y_var_subset, label, num_unique, c, m, alpha)
    else:
        ax.plot(sorted_x_var_subset, sorted_y_var_subset, label=label, color=c, marker=m, alpha=alpha)
        # ax.scatter(sorted_x_var_subset, sorted_y_var_subset, color=c, marker=m, alpha=alpha)


def plot_x_vs_y_with_errorbars(ax, sorted_x_var_subset, sorted_y_var_subset, label, num_unique, c, m, alpha):

    unique_sorted_x_var_subset = np.unique(sorted_x_var_subset)
    midpoints = np.zeros(num_unique)
    low_error = np.zeros(num_unique)
    high_error = np.zeros(num_unique)
    for i, u in enumerate(unique_sorted_x_var_subset):

        u_idxs = (sorted_x_var_subset == u)
        y_vals_for_u = sorted_y_var_subset[u_idxs]
        sorted_y_vals_for_u = sorted(y_vals_for_u)

        midpoints[i] = sorted_y_vals_for_u[int(len(sorted_y_vals_for_u) / 2)]
        low_error[i] = sorted_y_vals_for_u[0]
        high_error[i] = sorted_y_vals_for_u[-1]

    ax.plot(unique_sorted_x_var_subset, midpoints, label=label, color=c, marker=m, alpha=alpha)
    # ax.scatter(unique_sorted_x_var_subset, midpoints, color=c, marker=m, alpha=alpha)
    ax.fill_between(unique_sorted_x_var_subset, low_error, high_error, color=c, alpha=0.5)


def create_metrics_dict(configs):
    """load the relevant quantities for plotting that are stored in the json config files"""

    # loop through the config files and collect together various metrics
    METRIC_NAMES = ["total_num_ratios",
                    "true_mutual_info",
                    "prenormalised_kl",
                    "dv_bound",
                    "nwj_bound",
                    "ais_kl",
                    "ais_weight_vars",
                    "raise_kl",
                    "true_js",
                    "prenormalised_js",
                    "ais_js",
                    "direct_gauss_mse",
                    "indirect_gauss_mse",
                    "direct_gauss_nonzero_mse",
                    "indirect_gauss_nonzero_mse",
                    "direct_gauss_kl",
                    "indirect_gauss_kl",
                    "loss_function",
                    "shuffle_waymarks",
                    "head_type",
                    "objective_nu",
                    "n_batch",
                    "network_type",
                    "n_dims",
                    "n_imgs",
                    "waymark_mixing_increment",
                    "representation_learning_train_acc",
                    "representation_learning_test_acc",
                    ]

    metrics_dict = {name: [] for name in METRIC_NAMES}
    for config in configs:
        for name in METRIC_NAMES:
            if name in metrics_dict.keys():

                if name in config.keys():
                    metrics_dict[name].append(config[name])

                elif name == "n_imgs" and "n_imgs" in config["data_args"]:
                    metrics_dict[name].append(config["data_args"][name])

                else:
                    metrics_dict.pop(name)

    # convert metrics into arrays
    for key, val in metrics_dict.items():
        metrics_dict[key] = np.array(val)

    return metrics_dict


def plot_parameters_shared_vs_nonshared(comparison_save_dir, metrics_dict, x_var_name, y_var_names):
    shared_idxs = np.where(metrics_dict["shared_params"])[0]
    non_shared_idxs = np.where(np.logical_not(metrics_dict["shared_params"]))[0]

    curve_labels = ["shared params", "non-shared params"]
    curve_idxs = [shared_idxs, non_shared_idxs]

    plot_model_comparison(metrics_dict, x_var_name, y_var_names, curve_idxs, curve_labels, comparison_save_dir,
                          "parameter_sharing_mutual_info")


def plot_nwj_vs_nce(comparison_save_dir, metrics_dict, x_var_name, y_var_names):
    is_nwj = metrics_dict["loss_function"] == "nwj"
    nwj_idxs = np.where(is_nwj)[0]
    nce_idxs = np.where(np.logical_not(is_nwj))[0]

    curve_labels = ["variational KL loss", "variational JS loss"]
    curve_idxs = [nwj_idxs, nce_idxs]

    plot_model_comparison(metrics_dict, x_var_name, y_var_names, curve_idxs, curve_labels, comparison_save_dir,
                          "loss_function_choice_mutual_info")


def plot_shuffled_vs_nonshuffled(comparison_save_dir, metrics_dict, x_var_name, y_var_names):
    is_shuffled = metrics_dict["shuffle_waymarks"]
    shuffled_idxs = np.where(is_shuffled)[0]
    unshuffled_idxs = np.where(np.logical_not(is_shuffled))[0]

    curve_labels = ["shuffled", "unshuffled"]
    curve_idxs = [shuffled_idxs, unshuffled_idxs]

    plot_model_comparison(metrics_dict, x_var_name, y_var_names, curve_idxs, curve_labels, comparison_save_dir,
                          "shuffle_choice_mutual_info")


def plot_head_types(comparison_save_dir, metrics_dict, x_var_name, y_var_names):
    is_linear = metrics_dict["head_type"] == "linear"
    linear_idxs = np.where(is_linear)[0]
    mlp_idxs = np.where(np.logical_not(is_linear))[0]  # not linear implies mlp

    curve_labels = ["linear", "mlp"]
    curve_idxs = [linear_idxs, mlp_idxs]

    plot_model_comparison(metrics_dict, x_var_name, y_var_names, curve_idxs, curve_labels, comparison_save_dir,
                          "head_type_mutual_info")


def plot_batch_sizes(comparison_save_dir, metrics_dict, x_var_name, y_var_names):
    large_batch = metrics_dict["n_batch"] == 512
    large_batch_idxs = np.where(large_batch)[0]
    small_batch_idxs = np.where(np.logical_not(large_batch))[0]

    curve_labels = ["n_batch=512", "n_batch=128"]
    curve_idxs = [large_batch_idxs, small_batch_idxs]

    plot_model_comparison(metrics_dict, x_var_name, y_var_names, curve_idxs, curve_labels, comparison_save_dir,
                          "batch_size_mutual_info")


def plot_gauss_vs_mlp(comparison_save_dir, metrics_dict, x_var_name, y_var_names):
    is_gauss = metrics_dict["network_type"] == "quadratic"
    gauss_idxs = np.where(is_gauss)[0]
    mlp_idxs = np.where(np.logical_not(is_gauss))[0]

    curve_labels = ["quadratic", "mlp"]
    curve_idxs = [gauss_idxs, mlp_idxs]

    plot_model_comparison(metrics_dict, x_var_name, y_var_names, curve_idxs,
                          curve_labels, comparison_save_dir, "quadratic_vs_mlp_mutual_info")


def tre_vs_one_ratio(comparison_save_dir, metrics_dict, x_var_name, y_var_names, ylabels):

    is_tre = metrics_dict["total_num_ratios"] != 1
    is_one = metrics_dict["total_num_ratios"] == 1

    curve_labels = ["TRE", "single ratio"]
    curve_idxs = [is_tre, is_one]
    colors = ["red", "blue"]

    plot_model_comparison(metrics_dict, x_var_name, y_var_names, curve_idxs, curve_labels, comparison_save_dir,
                          "", xlabel="number of dimensions", ylabels=ylabels, curve_colors=colors)


def many_vs_one_ratio_multiomniglot(comparison_save_dir, metrics_dict, mlp):

    x_var_name = "n_imgs"
    xlabel = "number of characters"

    y_var_names = ["prenormalised_kl", "representation_learning_train_acc"]
    ylabels = ["mutual information", "mean label accuracy (train)"]

    n_imgs = metrics_dict["n_imgs"]
    mix_increments = metrics_dict["waymark_mixing_increment"]

    is_many = np.logical_or(n_imgs != mix_increments, n_imgs == 1)
    is_one = (n_imgs == mix_increments)

    many_idxs = np.where(is_many)[0]
    one_idxs = np.where(is_one)[0]

    curve_labels = ["1 ratio", "TRE"]
    curve_idxs = [one_idxs, many_idxs]
    colors = ["blue", "red"]
    markers = ["o", "^"]

    plot_model_comparison(metrics_dict, x_var_name, y_var_names, curve_idxs, curve_labels, comparison_save_dir,
                          "tre_vs_single_ratio_mi", xlabel=xlabel, ylabels=ylabels, curve_colors=colors, markers=markers)

    y_var_names = ["representation_learning_test_acc"]
    ylabels = ["mean label accuracy (test)"]
    if mlp:
        # 0.980, 0.979, 0.966, 0.919 0.828, 0.752, 0.709, 0.654, 0.581
        extend_metric([0.980, 0.919, 0.581], x_var_name, "representation_learning_test_acc",
                      curve_idxs, curve_labels, "WPC", metrics_dict)

        # 0.964, 0.949, 0.705, 0.467, 0.352, 0.278, 0.214, 0.178, 0.135
        extend_metric([0.964, 0.467, 0.135], x_var_name, "representation_learning_test_acc",
                      curve_idxs, curve_labels, "CPC", metrics_dict)
    else:
        # 1.0, 0.999, 0.997, 0.989, 0.904, 0.794
        extend_metric([1.0, 0.989, 0.794], x_var_name, "representation_learning_test_acc",
                      curve_idxs, curve_labels, "WPC", metrics_dict)

        # 0.986, 0.999, 0.998, 0.976, 0.847, 0.674
        extend_metric([0.986, 0.976, 0.674], x_var_name, "representation_learning_test_acc",
                      curve_idxs, curve_labels, "CPC", metrics_dict)

    colors.extend(['orange', 'green'])
    markers.extend(["s", "X"])
    plot_model_comparison(metrics_dict, x_var_name, y_var_names, curve_idxs, curve_labels, comparison_save_dir,
                          "tre_vs_single_ratio_linear_classification_acc", xlabel=xlabel, ylabels=ylabels,
                          curve_colors=colors, markers=markers)


def extend_metric(extend_vals, x_var_name, y_var_name, curve_idxs, curve_labels, label, metrics_dict):

    metrics_dict[x_var_name] = np.array(list(metrics_dict[x_var_name]) + [1, 4, 9])
    metrics_dict[y_var_name] = np.array(list(metrics_dict[y_var_name]) + extend_vals)
    curve_labels.append(label)
    start_idx = len(metrics_dict[x_var_name]) - len(extend_vals)
    curve_idxs.append([start_idx + i for i in range(len(extend_vals))])


def plot_nu_mutual_info(comparison_save_dir, metrics_dict, x_var_name, y_var_names):
    curve_labels = ["nce loss"]
    curve_idxs = [np.arange(len(metrics_dict["objective_nu"]))]

    plot_model_comparison(metrics_dict, x_var_name, y_var_names, curve_idxs, curve_labels, comparison_save_dir, "nu_mutual_info")


def create_xy_plots(args, comparison_save_dir, metrics_dict):
    # create plots of x_var against various y_var
    x_var_name = "total_num_ratios"
    y_var_names = ["prenormalised_kl"]
    y_labels = ["estimated mutual information"]

    if args.experiment_name == "parameter_sharing_mutual_info":
        plot_parameters_shared_vs_nonshared(comparison_save_dir, metrics_dict, x_var_name, y_var_names)

    elif args.experiment_name == "nwj_vs_nce_mutual_info":
        plot_nwj_vs_nce(comparison_save_dir, metrics_dict, x_var_name, y_var_names)

    elif args.experiment_name == "shuffled_vs_unshuffled_mutual_info":
        plot_shuffled_vs_nonshuffled(comparison_save_dir, metrics_dict, x_var_name, y_var_names)

    elif args.experiment_name == "linear_vs_mlp_heads_mutual_info":
        plot_head_types(comparison_save_dir, metrics_dict, x_var_name, y_var_names)

    elif args.experiment_name == "batch_size_mutual_info":
        plot_batch_sizes(comparison_save_dir, metrics_dict, x_var_name, y_var_names)

    elif args.experiment_name == "gauss_vs_mlp_bridges":
        plot_gauss_vs_mlp(comparison_save_dir, metrics_dict, x_var_name, y_var_names)

    if args.experiment_name == "many_vs_one_ratio_n_dims_mutual_info":
        tre_vs_one_ratio(comparison_save_dir, metrics_dict, x_var_name="n_dims", y_var_names=y_var_names, ylabels=y_labels)

    if args.experiment_name == "nu_mutual_info":
        plot_nu_mutual_info(comparison_save_dir, metrics_dict, x_var_name="objective_nu", y_var_names=y_var_names)

    if "multiomniglot_n_imgs" in args.experiment_name:
        many_vs_one_ratio_multiomniglot(comparison_save_dir, metrics_dict, mlp="mlp" in args.experiment_name)


def get_model_dirs_and_configs(dataset_name, model_timestamps):
    model_dirs = [os.path.join(project_root, "saved_models/{}/{}/".format(dataset_name, t)) for t in model_timestamps]
    configs = []
    for md in model_dirs:
        with open(md + "config.json") as f:
            configs.append(AttrDict(json.load(f)))
    return configs, model_dirs


def plot_logdiffs_for_datasplit(config, wmark_logprobs_path, md, which_set):
    data_logdiffs, consec_logdiffs, norm_consts, do_plot_refit_flows = get_wmark_logprobs(config, wmark_logprobs_path, which_set)
    plot_logdiffs(data_logdiffs, norm_consts, md + "figs/{}/".format(which_set), "data", do_plot_refit_flows)
    plot_logdiffs(consec_logdiffs, norm_consts, md + "figs/{}/".format(which_set), "parent_waymark", do_plot_refit_flows)


def get_wmark_logprobs(config, wmark_logprobs_path, which_set):
    num_flows = len(config.all_waymark_idxs)
    num_data = len(np.load(wmark_logprobs_path + "{}/wmark0_logprobs.npz".format(which_set))["bridge_wrt_data_0"])

    data_logdiffs = [np.zeros((num_flows - 1, num_data)), np.zeros((num_flows - 1, num_data)), np.zeros((num_flows - 1, num_data))]
    consec_logdiffs = [np.zeros((num_flows - 1, num_data, 2)), np.zeros((num_flows - 1, num_data)), np.zeros((num_flows - 1, num_data))]

    for i, wmark_idx in enumerate(config.all_waymark_idxs):
        load_obj = np.load(wmark_logprobs_path + "{}/wmark{}_logprobs.npz".format(which_set, wmark_idx))

        update_logdiff_arrays(data_logdiffs, load_obj, i, "data_{}".format(wmark_idx))
        update_logdiff_arrays(consec_logdiffs, load_obj, i, "cur_wmark_{}".format(wmark_idx))
        do_plot_refit_flows = update_logdiff_arrays(consec_logdiffs, load_obj, i, "prev_wmark_{}".format(wmark_idx))

    normalising_constants = log_mean_exp(consec_logdiffs[0][:, :, 1], axis=1)  # (n_ratios,)
    consec_logdiffs = [consec_logdiffs[0][:, :, 0], consec_logdiffs[1], consec_logdiffs[2]]

    return data_logdiffs, consec_logdiffs, normalising_constants, do_plot_refit_flows


def update_logdiff_arrays(logdiff_arrays, load_obj, i, name):
    _update_logdiff_array_helper(logdiff_arrays[0], load_obj, name, "bridge", i)
    _update_logdiff_array_helper(logdiff_arrays[1], load_obj, name, "wmark", i)
    try:
        _update_logdiff_array_helper(logdiff_arrays[2], load_obj, name, "refit_wmark", i)
        return True
    except:
        return False


def _update_logdiff_array_helper(logdiff_array, load_obj, name, mode, i):
    logger = logging.getLogger("tf")
    update_array = load_obj["{}_wrt_{}".format(mode, name)]

    if mode == "bridge" and i > 0:
        if "data" in name:
            logdiff_array[i - 1, :] = update_array
        elif "prev" in name:
            logdiff_array[i - 1, :, 0] = update_array
        else:
            logdiff_array[i - 1, :, 1] = update_array

    elif "wmark" in mode:
        if "data" in name:
            if i < len(logdiff_array):
                logdiff_array[i, :] += update_array
            if i > 0:
                logdiff_array[i - 1, :] -= update_array

            if i == 0:
                logger.info("av log prob of data dist wrt data: {}".format(np.mean(update_array)))
            if i == len(logdiff_array):
                logger.info("av log prob of final waymark dist wrt data: {}".format(np.mean(update_array)))

        elif "cur" in name and i < len(logdiff_array):
            logdiff_array[i, :] += update_array

        elif "prev" in name and i > 0:
            logdiff_array[i - 1, :] -= update_array


def plot_logdiffs(logdiffs, norm_consts, diff_save_dir, data_or_waymarks, do_plot_refit_flows):
    num_ratios = len(norm_consts)
    bridge_diffs, wmark_diffs, refit_wmark_diffs = logdiffs[0], logdiffs[1], logdiffs[2]  # [n_ratios, n]

    combined_logdiff_hists(bridge_diffs, wmark_diffs, refit_wmark_diffs, norm_consts,
                           num_ratios, data_or_waymarks, diff_save_dir, do_plot_refit_flows)

    separate_logdiff_hists(bridge_diffs, wmark_diffs, refit_wmark_diffs, norm_consts, num_ratios,
                           data_or_waymarks, diff_save_dir + "separate_hists/")

    # plot average logdiffs as a function of ratio idx
    plot_average_logdiffs_per_ratio(bridge_diffs, wmark_diffs, refit_wmark_diffs, norm_consts, num_ratios, diff_save_dir, data_or_waymarks)

    if data_or_waymarks == "data":
        save_true_vs_estimated_kl(bridge_diffs, wmark_diffs, refit_wmark_diffs, norm_consts, diff_save_dir)


def save_true_vs_estimated_kl(bridge_diffs, flow_diffs, refit_flow_diffs, norm_consts, diff_save_dir):
    logger = logging.getLogger("tf")

    true_kl = np.sum(np.mean(flow_diffs, axis=1))
    refit_flow_kl = np.sum(np.mean(refit_flow_diffs, axis=1))
    prenorm_estimated_kl = np.sum(np.mean(bridge_diffs, axis=1))
    estimated_kl = np.sum(np.mean(bridge_diffs, axis=1) - norm_consts)

    logger.info("True KL between data and noise dist: {}".format(true_kl))
    logger.info("Estimated KL between data and noise dist via refitting flows: {}".format(refit_flow_kl))
    logger.info("TRE Prenormalised estimated KL between data and noise dist: {}".format(prenorm_estimated_kl))
    logger.info("TRE Estimated KL between data and noise dist: {}".format(estimated_kl))

    np.savetxt(os.path.join(diff_save_dir, "true_vs_estimated_kl"),
               np.array([true_kl, refit_flow_kl, prenorm_estimated_kl, estimated_kl]),
               header="true_kl/refit_flow_kl/prenorm_estimated_kl/estimated_kl")


def plot_average_logdiffs_per_ratio(bridge_diffs, flow_diffs, refit_flow_diffs, norm_consts, num_ratios, diff_save_dir, data_or_waymarks):
    av_bridge_diffs = np.mean(bridge_diffs, axis=1)  # (n_ratios,)
    av_flow_diffs = np.mean(flow_diffs, axis=1)  # (n_ratios,)
    av_refit_flow_diffs = np.mean(refit_flow_diffs, axis=1)  # (n_ratios,)

    bridge_std_errors_on_means = np.std(bridge_diffs, axis=1) / (bridge_diffs.shape[1])**0.5
    flow_std_errors_on_means = np.std(flow_diffs, axis=1) / (bridge_diffs.shape[1])**0.5
    refit_flow_std_errors_on_means = np.std(refit_flow_diffs, axis=1) / (bridge_diffs.shape[1])**0.5

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(num_ratios), av_bridge_diffs, c='r', alpha=0.2)
    ax.scatter(np.arange(num_ratios), av_bridge_diffs, c='r', alpha=0.2, s=2)

    ax.errorbar(np.arange(num_ratios), av_bridge_diffs - norm_consts, yerr=bridge_std_errors_on_means, c='r', markeredgewidth=2, capsize=2)
    ax.scatter(np.arange(num_ratios), av_bridge_diffs - norm_consts, c='r', s=2)

    ax.errorbar(np.arange(num_ratios), av_flow_diffs, yerr=flow_std_errors_on_means, c='b', markeredgewidth=2, capsize=2)
    ax.scatter(np.arange(num_ratios), av_flow_diffs, c='b', s=2)

    ax.errorbar(np.arange(num_ratios), av_refit_flow_diffs, yerr=refit_flow_std_errors_on_means, c='g', markeredgewidth=2, capsize=2)
    ax.scatter(np.arange(num_ratios), av_refit_flow_diffs, c='g', s=2)

    save_fig(diff_save_dir, "av_logdiff_at_{}".format(data_or_waymarks))


def combined_logdiff_hists(bridge_diffs, flow_diffs, refit_flow_diffs, norm_consts, num_ratios, data_or_waymarks, diff_save_dir, do_plot_refit_flows):
    if do_plot_refit_flows:
        fig, axs = plt.subplots(num_ratios, 3)
    else:
        fig, axs = plt.subplots(num_ratios, 2)

    if num_ratios == 1: axs = np.array([axs])
    for i, sub_axs in enumerate(axs):
        if do_plot_refit_flows:
            ax1, ax2, ax3 = sub_axs
        else:
            ax1, ax3 = sub_axs
        plot_single_logdiff_hist(ax1, bridge_diffs, flow_diffs, refit_flow_diffs, norm_consts, i)
        if do_plot_refit_flows: plot_scatter_with_xyline(ax2, flow_diffs[i], refit_flow_diffs[i])
        plot_scatter_with_xyline(ax3, flow_diffs[i], bridge_diffs[i] - norm_consts[i])

    for ax in axs.ravel():
        ax.tick_params(axis='both', which='both', labelsize=7)
        for tick in ax.get_xticklabels():
            tick.set_visible(True)

    # fig.tight_layout()
    save_fig(diff_save_dir, "log_diff_at_{}_hists".format(data_or_waymarks))


def plot_scatter_with_xyline(ax, diffs1, diffs2):
    ax.scatter(diffs1, diffs2, alpha=0.5, c='g', s=0.5)
    low = min(diffs1.min(), diffs2.min()) - 1
    high = max(diffs1.max(), diffs2.max()) + 1
    line = np.linspace(low, high, 128)
    ax.plot(line, line, c='k')


def separate_logdiff_hists(bridge_diffs, flow_diffs, refit_flow_diffs, norm_consts, num_ratios, data_or_waymarks, diff_save_dir):
    for i in range(num_ratios):
        fig, axs = plt.subplots(1, 3)

        plot_single_logdiff_hist(axs[0], bridge_diffs, flow_diffs, refit_flow_diffs, norm_consts, i)
        plot_scatter_with_xyline(axs[1], flow_diffs[i], refit_flow_diffs[i])
        plot_scatter_with_xyline(axs[2], flow_diffs[i], bridge_diffs[i]-norm_consts[i])

        save_fig(diff_save_dir, "logdiff_at_{}_for_ratio{}".format(data_or_waymarks, i))


def plot_single_logdiff_hist(ax, bridge_diffs, flow_diffs, refit_flow_diffs, norm_consts, i):
    plot_hist(bridge_diffs[i], 0.1, ax, 'r')
    plot_hist(bridge_diffs[i] - norm_consts[i], 0.5, ax, 'r')
    plot_hist(flow_diffs[i], 0.5, ax, 'b')
    plot_hist(refit_flow_diffs[i], 0.5, ax, 'g')


def load_and_plot_samples(model_dirs, configs):
    for model_dir, config, in zip(model_dirs, configs):
        train_dp, _ = load_data_providers_and_update_conf(config)
        ais_dir = os.path.join(model_dir, "ais/")
        for ais_subdir in [ais_dir + sub for sub in os.listdir(ais_dir)]:
            try:
                rel_chains_dir = [sub for sub in os.listdir(ais_subdir) if "post_ais_chains" in sub][0]
                chains_dir = os.path.join(ais_subdir, rel_chains_dir)
                npz_files = [os.path.join(chains_dir, f) for f in os.listdir(chains_dir) if ".npz" in f]

                num_chains = len(npz_files)
                sample_shape = np.load(npz_files[0])["samples"].shape

                chains = np.empty((sample_shape[0], num_chains, *sample_shape[1:]))
                for i in range(num_chains):
                    load_file = [f for f in npz_files if "{}x1000".format(i) in f][0]
                    chains[:, i, ...] = np.load(load_file)["samples"]

                plot_chains_main(chains, "_".join(rel_chains_dir.split("_")[:-3]), ais_subdir, train_dp, config)
                config.data_args["logit"] = False
                plot_chains_main(chains, "no_logit_" + "_".join(rel_chains_dir.split("_")[:-3]), ais_subdir, train_dp, config, vminmax=None)
            except:
                FileNotFoundError("No samples exist for {}".format(model_dir))


# noinspection PyUnresolvedReferences,PyTypeChecker
def main():
    """Plot comparisons of different TRE models"""
    make_logger()
    np.set_printoptions(precision=3)

    parser = ArgumentParser(description='Aggregate results of TRE training', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default="gaussians")
    parser.add_argument('--id', action='append', type=str,
                        help="model timestamps plus config id",
                        default=[])
                        # default=["20200603-1152_0", "20200603-1152_1", "20200603-1152_2", "20200603-1152_5", "20200603-1152_6", "20200603-1152_7"])
                        # default=["20200514-2010_0"])
    parser.add_argument('--experiment_name', type=str, default="many_vs_one_ratio_n_dims_mutual_info")
    parser.add_argument('--plot_samples', type=int, default=-1)  # -1 == False else True
    args = parser.parse_args()

    dataset_name = args.dataset
    model_timestamps = args.id

    time_id = strftime('%Y%m%d-%H%M', gmtime())
    comparison_save_dir = os.path.join(project_root, "model_comparisons/{}/{}".format(dataset_name, time_id))
    os.makedirs(comparison_save_dir, exist_ok=True)

    model_ids_filename = os.path.join(comparison_save_dir, "model_timestamps.txt")
    with open(model_ids_filename, 'w+') as f:
        f.write("\n".join(model_timestamps))

    configs, model_dirs = get_model_dirs_and_configs(dataset_name, model_timestamps)

    if args.experiment_name:
        metrics_dict = create_metrics_dict(configs)
        create_xy_plots(args, comparison_save_dir, metrics_dict)

    if args.plot_samples != -1:
        load_and_plot_samples(model_dirs, configs)

    for md, config in zip(model_dirs, configs):
        wmark_logprobs_path = md + "true_wmark_logprobs/"
        if os.path.isdir(wmark_logprobs_path):
            plot_logdiffs_for_datasplit(config, wmark_logprobs_path, md, "train")
            plot_logdiffs_for_datasplit(config, wmark_logprobs_path, md, "val")


if __name__ == "__main__":
    main()
