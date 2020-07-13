from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
from utils.misc_utils import *
from utils.experiment_utils import *
from utils.plot_utils import *
from collections import OrderedDict
from __init__ import local_pc_root


# noinspection PyUnresolvedReferences
def eval_model(config):
    load_and_summarise_metrics("train", config)
    load_and_summarise_metrics("val", config)


# noinspection PyUnresolvedReferences
def plot_metrics_per_energy(stats, labels, dir_name, name):
    """for each NCE classifier, plot loss & classification accuracies"""
    n_stats = len(stats)
    # for each ratio, plot classification accs & loss
    fig, axs = plt.subplots(n_stats, 1)
    axs = axs.ravel() if isinstance(axs, np.ndarray) else [axs]
    x_axis = np.arange(len(stats[0]))
    for i, ax in enumerate(axs):
        ax.plot(x_axis, stats[i], alpha=0.5)
        ax.scatter(x_axis, stats[i], label=labels[i], marker='x')
        ax.set_xticks(np.arange(x_axis[0], x_axis[-1], 1.0))
        ax.legend()
    save_fig(dir_name, name)


# noinspection PyUnresolvedReferences
def load_metrics(metrics_dict, train_or_val, config):
    """load a list of metrics calculated during learning.

    Return a list of metrics, where each element of the list is an array
    whose length equals the number of ratios estimated in the ith parallelized job
    """

    load_dir = get_metrics_data_dir(save_dir, epoch_i=config.epoch_id)
    load_dir = os.path.join(load_dir, "{}.npz".format(train_or_val))
    metrics = np.load(load_dir)
    for key in metrics_dict.keys():
        if len(metrics[key].shape) >= 2:
            m = np.squeeze(metrics[key])
        else:
            m = metrics[key]
        metrics_dict[key].append(m)

    try:
        res = [np.concatenate(m) for m in metrics_dict.values()]
    except:
        res = [np.concatenate([s.reshape(1) for s in m]) for m in metrics_dict.values()]
    return res


# noinspection PyUnresolvedReferences
def load_and_summarise_metrics(which_set, config):
    """load and combine various metrics from different ratio estimation problems. print and plot results.

    Return: float - sum over all neg energies (averaged over the data)
    """
    logger = logging.getLogger("tf")
    labels = ["overall accuracy",
              "class1 accuracy",
              "class2 accuracy",
              "dawid_statistic",
              "loss",
              "nwj_loss",
              "av. neg-energy"]

    metrics_dict = OrderedDict([("acc", []),
                                ("class1_acc", []),
                                ("class2_acc", []),
                                ("dawid_statistic", []),
                                ("loss", []),
                                ("nwj_loss", []),
                                ("energy", [])])

    all_metrics = load_metrics(metrics_dict, which_set, config)

    plot_metrics_per_energy(all_metrics[:-1], labels[:-1], agg_save_dir, "{}_tre_classifier_stats".format(which_set))
    plot_metrics_per_energy(all_metrics[-1:], labels[-1:], agg_save_dir, "{}_av_energies".format(which_set))

    av_loss = np.mean(all_metrics[-3])
    total_neg_e = np.sum(all_metrics[-1])

    logger.info("-------------{} set-------------".format(which_set))
    logger.info("average loss: {}".format(av_loss))
    logger.info("total neg energy: {}".format(total_neg_e))

    save_energies_and_losses_to_txt(all_metrics, which_set)

    return total_neg_e


def save_energies_and_losses_to_txt(all_metrics, which_set):
    energies, losses = all_metrics[-1], all_metrics[-3]
    energies = np.concatenate([np.expand_dims(np.sum(energies), axis=0), energies])  # prepend the sum of all energies
    losses = np.concatenate([np.expand_dims(np.mean(losses), axis=0), losses])  # prepend the average of all losses

    np.savetxt(path_join(agg_save_dir, "{}_energies.txt".format(which_set)), energies, fmt='%10.2f', newline='')
    np.savetxt(path_join(agg_save_dir, "{}_losses.txt".format(which_set)), losses, fmt='%10.2f', newline='')


# noinspection PyUnresolvedReferences
def plot_learning_curves():
    """Plot learning curve for *each* bridge in TRE"""

    # 'trn_epoch_metrics' is dict whose vals are of shape (n_epochs, num_ratios)
    trn_epoch_metrics = get_per_epoch_losses("train")
    val_epoch_metrics = get_per_epoch_losses("val")

    for key in trn_epoch_metrics.keys():
        _plot_lr_curve_for_single_metric(trn_epoch_metrics[key], val_epoch_metrics[key], metric_name=key)


def _plot_lr_curve_for_single_metric(trn_metric, val_metric, metric_name):

    num_ratios = trn_metric.shape[1]

    all_ratios_fig, all_ratios_axs = plt.subplots(int(np.ceil(num_ratios ** 0.5)), int(np.ceil(num_ratios ** 0.5)))
    all_ratios_axs = all_ratios_axs.ravel() if isinstance(all_ratios_axs, np.ndarray) else [all_ratios_axs]

    average_trn_curve = np.zeros(trn_metric.shape[0])
    average_val_curve = np.zeros(val_metric.shape[0])

    for i in range(num_ratios):
        fig, ax = plt.subplots(1, 1)
        trn_curve = trn_metric[:, i]  # (n_epochs, )
        val_curve = val_metric[:, i]  # (n_epochs, )

        average_trn_curve += trn_curve
        average_val_curve += val_curve

        _plot_curves_helper(ax, fig, i, trn_curve, val_curve, metric_name=metric_name)
        _plot_curves_helper(all_ratios_axs[i], all_ratios_fig, i, trn_curve, val_curve, save=False)

    fig, ax = plt.subplots(1, 1)
    average_trn_curve /= num_ratios
    average_val_curve /= num_ratios
    _plot_curves_helper(ax, fig, "combined", average_trn_curve, average_val_curve, metric_name=metric_name)

    all_ratios_fig.tight_layout()
    remove_repeated_legends(all_ratios_fig)
    learning_curve_save_dir = os.path.join(agg_save_dir, "learning_curves/{}".format(metric_name))
    save_fig(learning_curve_save_dir, "all_ratios", all_ratios_fig)


def _plot_curves_helper(ax, fig, ratio_idx, trn_learning_curve, val_learning_curve, save=True, metric_name=""):

    ax.plot(np.arange(len(trn_learning_curve)), trn_learning_curve, label="train")
    ax.plot(np.arange(len(val_learning_curve)), val_learning_curve, label="val")
    if save:
        if metric_name == "tre_loss": ax.set_ylim((0, 2*np.log(2)))

        ax.set_xlabel("num epochs")
        ax.set_ylabel("loss")
        fig.legend()
        learning_curve_save_dir = os.path.join(agg_save_dir, "learning_curves/{}".format(metric_name))
        save_fig(learning_curve_save_dir, "ratio_{}".format(ratio_idx))


# noinspection PyUnresolvedReferences
def get_per_epoch_losses(train_or_val):

    per_epoch_metrics = {
        "tre_loss": [],
        "wmark_spacing_loss": [],
        "wmark_coef_graph_vars": []
    }

    metric_dir = get_metrics_data_dir(save_dir)
    final_num_losses = get_final_num_losses(train_or_val)
    num_epochs = len(os.listdir(metric_dir)) - 1  # subtract one for `best' epoch directory

    for epoch_i in range(num_epochs):
        try:
            load_dir = get_metrics_data_dir(save_dir, epoch_i=epoch_i)
            load_dir = os.path.join(load_dir, "{}.npz".format(train_or_val))

            metrics_dict = np.load(load_dir)
            loss = metrics_dict["loss"]
            wmark_loss = metrics_dict.get("wmark_spacing_loss", None)
            wmark_coefs = metrics_dict.get("wmark_coeffs", None)

            if len(loss) != final_num_losses:
                # discard any epochs where we had fewer wmarks than at end of learning.
                # ideally, we would still plot the lr curves for such epochs, but it involves more work
                continue

            loss = np.squeeze(loss, axis=0) if loss.shape[0] == 1 else loss

            per_epoch_metrics["tre_loss"].append(loss)
            per_epoch_metrics["wmark_spacing_loss"].append(wmark_loss)
            per_epoch_metrics["wmark_coef_graph_vars"].append(wmark_coefs)

        except FileNotFoundError:
            continue

    arrayify_epoch_stats(per_epoch_metrics)

    return per_epoch_metrics


def arrayify_epoch_stats(per_epoch_stats):
    """convert values of epoch_stats dict into arrays (instead of list of lists)"""
    keys_to_del = []
    for key in per_epoch_stats.keys():
        if per_epoch_stats[key][0] is None:
            keys_to_del.append(key)
        else:
            per_epoch_stats[key] = np.array(per_epoch_stats[key])  # (n_epochs, n_losses)
            if len(per_epoch_stats[key].shape) == 1:
                per_epoch_stats[key] = np.expand_dims(per_epoch_stats[key], axis=1)

    for key in keys_to_del:
        del per_epoch_stats[key]


def get_final_num_losses(train_or_val):
    best_load_dir = get_metrics_data_dir(save_dir, "best")
    best_load_dir = os.path.join(best_load_dir, "{}.npz".format(train_or_val))
    final_num_losses = len(np.load(best_load_dir)["loss"])
    return final_num_losses


def make_global_config():
    """load & augment experiment configuration, then add it to global variables"""
    parser = ArgumentParser(description='Aggregate results of TRE training', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_path', type=str, default="gaussians/20200713-1029_4")
    parser.add_argument('--script_id', type=int, default=0)
    parser.add_argument('--epoch_id', type=str, default="best")
    args = parser.parse_args()

    with open(project_root + "saved_models/{}/config.json".format(args.config_path)) as f:
        config = json.load(f)

    rename_save_dir(config)
    config.update(vars(args))
    config["agg_save_dir"] = os.path.join(config["save_dir"], "aggregated_results/")
    save_config(config)

    return AttrDict(config)


# noinspection PyUnresolvedReferences,PyTypeChecker
def main():
    """Plot metrics for a trained TRE model, including the learning curves"""

    make_logger()
    np.set_printoptions(precision=3)

    # load a config file whose contents are added to globals(), making them easily accessible elsewhere
    config = make_global_config()
    train_dp, _ = load_data_providers_and_update_conf(config)
    globals().update(config)

    eval_model(config)
    plot_learning_curves()


if __name__ == "__main__":
    main()
