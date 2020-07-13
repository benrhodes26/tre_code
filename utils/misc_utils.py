import functools
import logging
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
from __init__ import local_pc_root, project_root
from time import time
from scipy.stats import norm
from scipy.special import logsumexp


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def one_hot_encode(labels, n_labels):
    """
    Transforms numeric labels to 1-hot encoded labels. Assumes numeric labels are in the range 0, 1, ..., n_labels-1.

    Credit to George Papamakarios (https://github.com/gpapamak/maf)
    """

    assert np.min(labels) >= 0 and np.max(labels) < n_labels

    y = np.zeros([labels.size, n_labels])
    y[np.arange(labels.size), labels] = 1

    return y


def logit(x):
    """
    Elementwise logit (inverse logistic sigmoid).

    :param x: numpy array
    :return: numpy array
    """
    return np.log(x / (1.0 - x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def log_mean_exp(x, axis=None):
    if axis is not None:
        ax_len = x.shape[axis]
    else:
        ax_len = x.size
    return logsumexp(x, axis=axis) - np.log(ax_len)


def log_var_exp(x):
    """Given x=log(w), compute log(var(w)) using numerically stable operations"""
    mu = log_mean_exp(x)
    x_max = np.max(x)
    x_prime = x - x_max
    mu_prime = mu - x_max
    summand = np.exp(2 * x_prime) - np.exp(2 * mu_prime)
    return 2 * x_max + np.log(np.sum(summand)) - np.log(len(x))


def five_stat_sum(x, axis=None):
    return np.array([x.mean(axis=axis), np.median(x, axis=axis), x.std(axis=axis), x.min(axis=axis), x.max(axis=axis)])


def five_stat_sum_v2(x, axis=None):
    means, stds = x.mean(axis=axis), x.std(axis=axis)
    return np.array([means, means - stds, means + stds, x.min(axis=axis), x.max(axis=axis)])


def five_stat_sum_v3(x, axis=None, quantile=0.99):
    return np.array([x.mean(axis=axis), np.quantile(x, 1-quantile, axis=axis),
                     np.quantile(x, quantile, axis=axis), x.min(axis=axis), x.max(axis=axis)])


def copytree(src, dst, filenames=None, symlinks=False, ignore=None):
    items = filenames if filenames is not None else os.listdir(src)
    for item in items:
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def rename_save_dir(config):
    """The save_dir path may need changing if we have copied files from cluster to local pc"""
    s_dir = config["save_dir"]
    l = s_dir.split("/")
    idx = l.index("saved_models")
    config["save_dir"] = project_root + "saved_models/" + "/".join(l[idx+1:])

    f_dir = config["fig_dir_name"]
    l = f_dir.split("/")
    idx = l.index("figs")
    config["fig_dir_name"] = project_root + "figs/" + "/".join(l[idx+1:])


def batched_operation(data, op, batch_size):
    n = data.shape[0]
    output = np.zeros_like(data)
    for j in range(0, n, batch_size):
        if j + batch_size < n:
            output[j:j + batch_size, ...] = op(data[j:j+batch_size, ...])
        else:
            output[j:, ...] = op(data[j:, ...])

    return output


def np_expand_multiple_axis(x, axes):
    axes = sorted(axes)
    for ax in axes[::-1]:
        x = np.expand_dims(x, axis=ax)
    return x


def cross_entropy_two_gaussians(A, B):
    """cross entropy H(p_A, p_B) where p_A ~ N(0, A), p_B ~ N(0, B)"""
    trace_term = np.trace(np.linalg.solve(B, A))
    const = len(B) * np.log(2 * np.pi)
    B_log_abs_det = np.linalg.slogdet(B)[1]
    cross_entropy = 0.5 * (trace_term + const + B_log_abs_det)

    return cross_entropy


def kl_between_two_gaussians(A, B):
    """D_kl(p_A, p_B) where p_A ~ N(0, A), p_B ~ N(0, B)"""
    trace_term = np.trace(np.linalg.solve(B, A))
    const = -len(B)
    B_log_abs_det = np.linalg.slogdet(B)[1]
    A_log_abs_det = np.linalg.slogdet(A)[1]
    diff_log_abs_det = B_log_abs_det - A_log_abs_det

    kl = 0.5 * (trace_term + const + diff_log_abs_det)

    return kl


def path_join(*x):
    return os.path.join(*x)


def convert_to_bits_per_dim(logp, d, scale=1.0):
    return (1/np.log(2)) * (np.log(scale) - (logp/d))
