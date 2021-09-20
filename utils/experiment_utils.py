import json
import logging
import numpy as np
import os

from scipy.special import logsumexp
from time import time

from __init__ import project_root
from data_handlers.data_providers import load_data_providers, DataProvider
from utils.plot_utils import disp_imdata
from utils.project_constants import IMG_DATASETS


def get_waymark_data_dir(seq_id, dataset_name, current=True):
    if current:
        return project_root + "waymark_data/{}/id{}/".format(dataset_name, seq_id)
    else:
        return project_root + "waymark_data/{}/old/id{}/".format(dataset_name, seq_id)


def get_metrics_data_dir(model_save_dir, epoch_i=None):
    if epoch_i is None:
        metrics_save_dir = os.path.join(model_save_dir, "metrics/")
    else:
        metrics_save_dir = os.path.join(model_save_dir, "metrics/epoch_{}/".format(epoch_i))
    os.makedirs(metrics_save_dir, exist_ok=True)
    return metrics_save_dir


def make_logger():
    logger = logging.getLogger("tf")
    logger.setLevel(logging.DEBUG)
    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Taken from https://github.com/cs230-stanford/cs230-code-examples/blob/master/tensorflow/nlp/model/utils.py
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


# noinspection PyUnresolvedReferences
def check_early_stopping(saver, sess, save_path, config):
    val_loss = config["current_val_loss"]
    best_val_loss = config["best_val_loss"]

    saved = False
    if val_loss and val_loss < best_val_loss:
        config.best_val_loss = val_loss
        config.n_epochs_until_stop = config.patience
        saver.save(sess, save_path)
        saved = True

    do_break = False
    val_loss_is_nan = val_loss and np.isnan(val_loss)
    if (config.n_epochs_until_stop <= 0) or val_loss_is_nan:
        do_break = True

    return do_break, saved


def update_time_avg(config, pre_sample_time, time_key, counter_key):
    """Update a running average of a time stored in config[time_key].

    This is useful for tracking the average time spent on an operation
    performed multiple times during learning e.g sampling noise once an epoch"""
    new_time = time() - pre_sample_time
    cur_mean_time, num = config.get(time_key, 0), config.get(counter_key, 0)
    config[time_key] = (num * cur_mean_time + new_time) / (num + 1)
    config[counter_key] = num + 1


def get_mcmc_intervals(mcmc_params):
    start, stop, num, thinning_factor = mcmc_params
    step_sizes = np.linspace(start, stop, num) ** 2
    return [[2, thinning_factor, s] for s in step_sizes]


def seperately_permute_matrix_cols(X, transpose=False):
    if transpose: X = X.T
    X = X[..., np.newaxis]  # (n, d, 1)
    X = DataProvider.independent_sliced_shuffle(X)  # (n, d, 1)
    X = np.squeeze(X, axis=-1)  # (n, d)
    if transpose: X = X.T
    return X


def save_config(config, save_dir=None):
    if not save_dir:
        save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # ensure that there are no np.floats, np.ints or np.ndarrays since they aren't serializable
    is_np_float = lambda x: isinstance(x, (np.float32, np.float64))
    is_np_int = lambda x: isinstance(x, (np.int32, np.int64))
    bad_keys = []
    for key, val in config.items():
        if is_np_float(val):
            config[key] = float(val)
        elif is_np_int(val):
            config[key] = int(val)
        elif isinstance(val, list) and is_np_float(val[0]):
            config[key] = [float(v) for v in val]
        elif isinstance(val, list) and is_np_int(val[0]):
            config[key] = [int(v) for v in val]
        elif isinstance(val, np.ndarray):  # don't save arrays
            bad_keys.append(key)

    for key in bad_keys:
        del config[key]

    with open(save_dir + "/config.json", 'w') as fp:
        json.dump(config, fp, indent=4)


# noinspection PyUnresolvedReferences
def load_data_providers_and_update_conf(config, include_test=False, dataset_name=None, shuffle=True, only_val=False, use_labels=False):

    dataset_name = config["dataset_name"] if dataset_name is None else dataset_name
    train_dp, val_dp, test_dp = load_data_providers(dataset_name,
                                                    config["n_batch"],
                                                    seed=config["data_seed"],
                                                    use_labels=use_labels,
                                                    frac=config["frac"],
                                                    shuffle=shuffle,
                                                    data_args=config["data_args"])

    config.update({"n_dims": int(train_dp.n_dims), "n_samples": int(train_dp.n_samples)})
    config.update({"n_val_samples": int(val_dp.n_samples)})
    config.update({"train_data_stds": np.std(train_dp.data, axis=0)})
    config.update({"train_data_min_max": [train_dp.data.min(), train_dp.data.max()]})
    print("n_train, n_val: {}, {}".format(train_dp.n_samples, val_dp.n_samples))

    if hasattr(train_dp.source, "cov_mat"):
        config.update({"cov_mat": train_dp.source.cov_mat})
    if hasattr(train_dp.source, "logit_alpha"):
        config.update({"logit_alpha": train_dp.source.logit_alpha})

    if hasattr(train_dp.source, "logit_shift"):
        config.update({"preprocess_logit_shift": train_dp.source.logit_shift})
    if hasattr(train_dp.source, "shift"):
        config.update({"preprocess_shift": train_dp.source.shift})

    if hasattr(train_dp, "labels") and train_dp.labels is not None:
        labels = train_dp.labels
        label_shape = labels.shape
        if len(label_shape) == 2:
            config["num_classification_problems"] = label_shape[1]
            num_classes_per_problem = np.array([len(np.unique(labels[:, i])) for i in range(label_shape[1])])
            config["num_classes_per_problem"] = num_classes_per_problem
            config["max_num_classes"] = np.max(num_classes_per_problem)
        else:
            config["num_classification_problems"] = 1
            config["max_num_classes"] = len(np.unique(labels))

        if config["dataset_name"] == "multiomniglot":
            config["true_mutual_info"] = np.sum(np.log(num_classes_per_problem))

    if only_val:
        return val_dp
    elif include_test:
        return train_dp, val_dp, test_dp
    else:
        return train_dp, val_dp


def dv_bound_fn(e1, e2):
    term1 = np.mean(e1)
    term2 = -logsumexp(e2) + np.log(len(e2))
    bound = term1 + term2
    return bound, term1, term2


def nwj_bound_fn(e1, e2):
    term1 = np.mean(e1) + 1
    term2 = - np.mean(np.exp(e2))
    bound = term1 + term2

    return bound, term1, term2


def log_sigmoid(x):
    return np.minimum(x, 0) - np.log(1 + np.exp(-np.abs(x)))


def np_nce_loss(neg_energy1, neg_energy2, nu=1):
    log_nu = np.log(nu)
    term1 = log_sigmoid(neg_energy1 - log_nu)  # (n, n_ratios)
    term2 = log_sigmoid(log_nu - neg_energy2)  # (n, n_ratios)

    loss_term1 = -np.mean(term1, axis=0)
    loss_term2 = -nu * np.mean(term2, axis=0)
    loss = loss_term1 + loss_term2   # (n_ratios, )

    return loss, loss_term1, loss_term2


def jensen_shannon_fn(e1, e2, logz):
    m1 = np.stack([e1, np.ones_like(e1)*logz], axis=1)
    m2 = np.stack([e2, np.ones_like(e2)*logz], axis=1)

    term1 = np.log(2) + np.mean(e1) - np.mean(logsumexp(m1, axis=1))
    term2 = np.log(2) + logz - np.mean(logsumexp(m2, axis=1))

    return 0.5 * (term1 + term2)


def plot_chains_main(chains, name, save_dir, dp, config, vminmax = (0, 1), max_n_states=10):

    if (config.dataset_name in IMG_DATASETS) or ("pca" in config.dataset_name):

        max_n_chains = 10000
        skip_n = max(1, int(len(chains) / max_n_chains))
        x = revert_data_preprocessing(chains[::skip_n], dp, is_wmark_input=True)

        layout_dim = 7
        n_pages = int(np.ceil(len(x) / layout_dim ** 2))

        disp_imdata(imgs=x[:, -1, ...],
                    dataset_name=config.dataset_name,
                    dir_name=os.path.join(save_dir, "{}_final_states/".format(name)),
                    layout=[layout_dim, layout_dim],
                    num_pages=n_pages,
                    vminmax=vminmax
                    )

        if chains.shape[1] > 1:
            # plot all states for individual chains
            n_chains_to_plot = min(50, len(x))
            n_chains_in_figure = 10
            n_figures = int(np.ceil(n_chains_to_plot / n_chains_in_figure))
            n_skip = int(np.ceil(x.shape[1] / min(max_n_states, x.shape[1])))
            n_states_in_figure = int(np.ceil(x.shape[1] / n_skip))
            chains_event_dims = x.shape[2:]

            for i in range(n_figures):
                disp_imdata(imgs=x[n_chains_in_figure * i:n_chains_in_figure * (i + 1), ::n_skip].reshape(-1, *chains_event_dims),
                            dataset_name=config.dataset_name,
                            dir_name=os.path.join(save_dir, "{}_whole_chains/".format(name)),
                            # layout=[int(np.ceil(len(chains[i]) ** 0.5))] * 2,
                            layout=[n_chains_in_figure, n_states_in_figure],
                            name=str(i),
                            vminmax=vminmax
                            )


def revert_data_preprocessing(data, dp, is_wmark_input):

    if is_wmark_input:
        n, k, event_dims = data.shape[0], data.shape[1], data.shape[2:]
        data = data.reshape((-1, *event_dims))  # (n*k, ...)

    data = dp.source.reverse_preprocessing(data)

    if is_wmark_input:
        event_dims = data.shape[1:]
        data = data.reshape(n, k, *event_dims)  # (n, k, ...)

    return data


def logit_inv(x, alpha):
    return (sigmoid(x) - alpha)/(1 - 2 * alpha)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
