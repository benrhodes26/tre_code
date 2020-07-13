import functools
import logging
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil

from __init__ import project_root, local_pc_root
from utils.project_constants import IMG_DATASETS
from time import time
from scipy.stats import norm

from collections import OrderedDict
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import FormatStrFormatter
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import (
    NullFormatter, ScalarFormatter, LogFormatterSciNotation, LogitFormatter,
    NullLocator, LogLocator, AutoLocator, AutoMinorLocator,
    SymmetricalLogLocator, LogitLocator)

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIG_SIZE = 14
HUGE_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=HUGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=HUGE_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIG_SIZE)    # legend fontsize
plt.rc('figure', titlesize=18)  # fontsize of the figure title

# rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']


def reset_all_fontsizes():
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIG_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIG_SIZE)  # fontsize of the figure title


def set_all_fontsizes(size):
    plt.rc('font', size=size)  # controls default text sizes
    plt.rc('axes', titlesize=size)  # fontsize of the axes title
    plt.rc('axes', labelsize=size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=size)  # legend fontsize
    plt.rc('figure', titlesize=size)  # fontsize of the figure title


def remove_repeated_legends(fig):
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


def plot_hist(vals, alpha = 1.0, ax = None, color = None, label = None, x_range = None, y_lim = None, dir_name=None, name=None):
    if x_range and x_range == "exclude_outliers":
        x_mean, x_std = vals.mean(), vals.std()
        vals = vals[np.abs(vals - x_mean) < 3 * x_std]

    if ax is None:
        save = True
        fig, ax = plt.subplots(1, 1)
    else:
        save = False

    h, _, _ = ax.hist(vals,
                      density=True,
                      alpha=alpha,
                      color=color,
                      label=label,
                      bins=int(len(vals) ** 0.5),
                      )

    if y_lim is not None:
        ax.set_ylim(y_lim)

    if save:
        save_fig(dir_name, name)

    return h


def disp_imdata(imgs, dataset_name, num_pages=1, imsize=None, layout=None, dir_name=None, name="", vminmax=(0, 1)):
    """
    Displays multiple pages of image data
    :param imgs: an numpy array with images as rows
    :param imsize: size of the images
    :param layout: layout of images in a page
    :return: none
    """
    greyscale = False
    if imsize is None:
        if len(imgs.shape) == 4:
            if imgs.shape[-1] == 1:
                greyscale = True
                imsize = imgs.shape[-3:-1]
            else:
                imsize = imgs.shape[1:]
        elif len(imgs.shape) == 2:
            img_dim = imgs.shape[-1]
            imsize = [int(img_dim**0.5)]*2
            greyscale = True
        else:
            raise ValueError("do not know how to infer img shape for data of shape {}".format(imgs.shape))

    if layout is None:
        layout = [7, 7]

    if isinstance(layout, int): layout = [layout, 1]

    num_imgs_per_page = np.prod(np.array(layout))

    for use_minmax in [False, True]:
        for i in range(num_pages):
            fig, axs = plt.subplots(layout[0], layout[1])

            if isinstance(axs, np.ndarray):
                axs = axs.flatten()
            else:
                axs = [axs]
            for ax in axs:
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)

            imgs_i = imgs[i*num_imgs_per_page: (i+1)*num_imgs_per_page]
            for j, im in enumerate(imgs_i):

                if greyscale:
                    if use_minmax:
                        axs[j].imshow(im.reshape(imsize), cmap='gray', interpolation='none', vmin=0, vmax=1)
                    else:
                        axs[j].imshow(im.reshape(imsize), cmap='gray', interpolation='none')
                else:
                    if use_minmax:
                        axs[j].imshow(im.reshape(imsize), vmin=0, vmax=1)
                    else:
                        axs[j].imshow(im.reshape(imsize))

            fig_name = name + "_page_{}".format(i)
            if use_minmax: fig_name += "_vminmax"
            save_fig(dir_name, fig_name)


def plot_wmark_hists(config, train_data, val_data):
    n_waymarks = train_data.shape[1]
    fig, axs = plt.subplots(1, n_waymarks)
    axs = axs.ravel()
    for i in range(n_waymarks):
        axs[i].hist(train_data[:, i, ...].flatten(), bins=int(np.sqrt(len(train_data[:, i, ...]))), alpha=0.5, color='r', density=True)
        axs[i].hist(val_data[:, i, ...].flatten(), bins=int(np.sqrt(len(val_data[:, i, ...]))), alpha=0.5, color='b', density=True)
    save_fig(config.save_dir, "data_hists")


def plot_hist_marginals_and_scatter(data,
                                    lims=None,
                                    gt=None,
                                    plot_standard_normal=False,
                                    axis_on=True,
                                    axs=None,
                                    labels=None,
                                    colours=None,
                                    alpha=0.5):
    """
    Plots marginal histograms and pairwise scatter plots of a dataset.
    """
    if not isinstance(data, list):
        data = [data]

    n_bins = int(np.sqrt(data[0].shape[0]))

    if data[0].ndim == 1:
        if axs is None:
            fig, axs = plt.subplots(1, 1)

        for d in data:
            axs.hist(d, n_bins, density=True)
        axs.set_ylim([0, axs.get_ylim()[1]])
        if lims is not None: axs.set_xlim(lims)
        if gt is not None: axs.vlines(gt, 0, axs.get_ylim()[1], color='r')

    else:
        n_dim = data[0].shape[1]
        if axs is None:
            fig, axs = plt.subplots(n_dim, n_dim)
            axs = np.array([[axs]]) if n_dim == 1 else axs

        if lims is not None:
            lims = np.asarray(lims)
            lims = np.tile(lims, [n_dim, 1]) if lims.ndim == 1 else lims

        for i in range(n_dim):
            for j in range(n_dim):
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

                for k, d in enumerate(data):
                    label = labels[k] if labels else ""
                    c = colours[k] if colours else None
                    if i == j:
                        axs[i, j].hist(d[:, i], n_bins, density=True, label=label, color=c, alpha=alpha)

                        if plot_standard_normal:
                            x_ax = np.arange(d[:, i].min() - 1, d[:, i].max() + 1, 0.001)
                            axs[i, j].plot(x_ax, norm.pdf(x_ax, 0, 1), linewidth=0.6)

                        axs[i, j].set_ylim([0, axs[i, j].get_ylim()[1]])
                        if lims is not None: axs[i, j].set_xlim(lims[i])
                        if gt is not None: axs[i, j].vlines(gt[i], 0, axs[i, j].get_ylim()[1], color='r')

                    else:
                        axs[i, j].plot(d[:, i], d[:, j], 'k.', ms=0.2, label=label, c=c, alpha=alpha)
                        if lims is not None:
                            axs[i, j].set_xlim(lims[i])
                            axs[i, j].set_ylim(lims[j])
                        if gt is not None: axs[i, j].plot(gt[i], gt[j], 'r.', ms=8)
    if axis_on:
        plt.axis('on')
    # plt.show(block=False)


def plot_hist_marginals(data, labels=None, colours=None, lim=None, gt=None):
    if not isinstance(data, list):
        data = [data]

    n_bins = int(np.sqrt(data[0].shape[0]))
    n_dim = data[0].shape[1]
    height = width = int(np.ceil(n_dim**0.5))
    fig, ax = plt.subplots(height, width, sharex=True, sharey=True)
    ax = ax.ravel()
    for i in range(n_dim):
        for j, d in enumerate(data):
            label = labels[j] if labels else ""
            colour = colours[j] if colours else None
            ax[i].hist(d[:, i], n_bins, density=True, alpha=0.5, label=label, color=colour)

        ax[i].set_ylim([0, ax[i].get_ylim()[1]])
        if lim is not None: ax[i].set_xlim(lim)
        if gt is not None: ax[i].vlines(gt[i], 0, ax[i].get_ylim()[1], color='r')


def plot_hists_for_each_dim(n_dims_to_plot, data, labels=None, colours=None, dir_name=None,
                            filename=None, increment=10, include_scatter=False, alpha=0.5):

    if not isinstance(data, list):
        data = [data]

    k = int(np.ceil(n_dims_to_plot / increment))
    for i in range(k):
        j = i * increment
        inputs = [d[:, j:j+increment] for d in data]

        if include_scatter:
            plot_hist_marginals_and_scatter(inputs, labels=labels, colours=colours, alpha=alpha)
        else:
            plot_hist_marginals(inputs, labels=labels, colours=colours)

        save_fig(dir_name, "{}_dims_{}_to_{}".format(filename, j, min(j+increment, n_dims_to_plot)))


def save_fig(dir_name, filename, fig=None, format="pdf", **args):
    """save figure as .pdf to project_root/figs/dir_name"""
    os.makedirs(dir_name, exist_ok=True)
    if fig is not None:
        fig.savefig(os.path.join(dir_name, "{}.{}".format(filename, format)), dpi=300, **args)
    else:
        plt.savefig(os.path.join(dir_name, "{}.{}".format(filename, format)), dpi=300, **args)
    plt.close()


def five_stat_and_hist(x, name, save_dir):
    np.savetxt(os.path.join(save_dir, "ratio_{}_5statsum.txt".format(name)),
               _five_stat_sum(x), header="mean/median/std/min/max")
    plot_hist_marginals_and_scatter(x)
    save_fig(save_dir, "ratio_{}_histogram".format(name))


def _five_stat_sum(x):
    return np.array([x.mean(), np.median(x), x.std(), x.min(), x.max()])


def mult_plotscatter_single_axis(ax, x, xlabel, ylabel, title=None, x_axis=None, labels=None):
    if len(x.shape) == 1:
        x = x.reshape(1, -1)

    if x_axis is None:
        x_axis = np.arange(x.shape[1])
    for i in range(x.shape[0]):
        l = labels[i] if labels is not None else None
        ax.plot(x_axis, x[i], label=l)
        ax.scatter(x_axis, x[i], s=5.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

def plotscatter_single_axis(ax, x, xlabel, ylabel, title=None, x_axis=None):
    if x_axis is None:
        x_axis = np.arange(x.shape[0])
    ax.plot(x_axis, x)
    ax.scatter(x_axis, x, s=5.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

# noinspection PyUnresolvedReferences
def plotscatter_one_per_axis(x, xlabels, ylabels, dir_name, name, title=None):
    """Plot each row of x in a separate subplot. Use both plt.plot & plt.scatter."""
    n_subplots = len(x)
    fig, axs = plt.subplots(n_subplots, 1)

    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    axs = axs.ravel()

    x_axis = np.arange(len(x[0]))
    for i, ax in enumerate(axs):

        ax.scatter(x_axis, x[i])
        ax.plot(x_axis, x[i])

        ax.set_xlabel(xlabels[i])
        ax.set_ylabel(ylabels[i])
        ax.set_xticks(np.arange(x_axis[0], x_axis[-1], 1.0))

    if title:
        fig.suptitle(title)
    save_fig(dir_name, name)

def custom_cmap_with_zero_included(vals):

    # define the colormap
    cmap = plt.get_cmap('bwr')
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins, normalize and force 0 to be part of the colorbar
    min_e, max_e = np.min(vals), np.max(vals)
    lim = max(np.abs(min_e), np.abs(max_e)) + 1
    try:
        bounds = np.arange(-lim, lim, lim/10)
    except:
        bounds = np.arange(-10, 10, 1)

    idx = np.searchsorted(bounds, 0)
    bounds = np.insert(bounds, idx, 0)
    norm = BoundaryNorm(bounds, cmap.N)

    return cmap, norm


def create_subplot_with_max_num_cols(n_figs, max_n_cols):
    n_rows = int(np.ceil(n_figs / max_n_cols))
    n_cols = n_figs if n_rows == 1 else max_n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axs = axs.ravel() if isinstance(axs, np.ndarray) else [axs]
    return axs, fig


class CustomScale(mscale.ScaleBase):
    name = 'custom'

    def __init__(self, axis, thresh=1e-4, **kwargs):
        mscale.ScaleBase.__init__(self, axis)
        self.base = 10
        self.thresh = thresh

    def get_transform(self):
        return self.CustomTransform(self.thresh)

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to specialized versions for
        log scaling.
        """
        axis.set_major_locator(LogLocator(self.base))
        axis.set_major_formatter(LogFormatterSciNotation(self.base))
        axis.set_minor_locator(LogLocator(self.base, None))
        axis.set_minor_formatter(LogFormatterSciNotation(self.base,labelOnlyBase=False))

    class CustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh

        def transform_non_affine(self, a):
            return np.log10(self.thresh + a)

        def inverted(self):
            return CustomScale.InvertedCustomTransform(self.thresh)

    class InvertedCustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, thresh):
            mtransforms.Transform.__init__(self)
            self.thresh = thresh

        def transform_non_affine(self, a):
            return 10**a - self.thresh

        def inverted(self):
            return CustomScale.CustomTransform(self.thresh)

# Now that the Scale class has been defined, it must be registered so
# that ``matplotlib`` can find it.
mscale.register_scale(CustomScale)
