import dill
from __init__ import density_data_root

from utils.misc_utils import *
from utils.experiment_utils import *
from utils.plot_utils import *


def plot_marginal(ax, x, kde=None):
    # Plot the histogram
    ax.hist(x, bins=100, density=True, label='Histogram from samples', zorder=5, edgecolor='k', alpha=0.5)

    if kde is not None:
        # Plot the KDE as fitted using the default arguments
        ax.plot(kde.support, kde.density, lw=3, label='KDE from samples', zorder=10)

    # Plot the samples
    ax.scatter(x, np.abs(np.random.randn(x.size)) / 40,
               marker='x', color='red', zorder=20, label='Samples', alpha=0.2)

    ax.legend(loc='best')
    ax.grid(True, zorder=-5)


def plot_all_marginals(data, kdes=None, save_dir=None, filename=None):
    n_dims = data.shape[1]
    n_subplots = 20
    n_figs = n_dims // n_subplots

    for i in range(n_figs):
        fig, axs = plt.subplots(int(np.ceil(n_subplots/2)), 2, figsize=(15, int((n_subplots/2) * 0.5 * 3)))
        axs = axs.ravel()
        for j in range(n_subplots):
            idx = i*n_subplots + j
            x = data[:, idx].astype(np.double)
            kde = kdes[idx] if kdes is not None else None
            plot_marginal(axs[j], x, kde)

        save_fig(save_dir, filename + "_{}".format(i))


def kde_of_marginals(data, bw_mult, plot=True):

    import statsmodels.api as sm

    n_dims = data.shape[1]
    if isinstance(bw_mult, float): bw_mult = np.array([bw_mult] * n_dims)

    if plot:
        fig, axs = plt.subplots(int(n_dims/2), 2, figsize=(15, int(n_dims*0.5*3)))
        axs = axs.ravel()

    kdes = []
    for i in range(n_dims):
        x = data[:, i].astype(np.double)
        kde = sm.nonparametric.KDEUnivariate(x)
        kde.fit(kernel="gau", fft=True, bw="silverman", adjust=bw_mult[i])  # Estimate the densities
        kdes.append(kde)

        if plot:
            plot_marginal(axs[i], x, kde)

    return kdes


def make_cdf_fn(supports, cdfs):

    # def enlarge_domains(x, sup, cdf):
    #     c = len(sup)
    #
    #     lower_s = min(x.min()-1, sup[0])
    #     upper_s = max(x.max()+1, sup[c-1])
    #
    #     sup = np.insert(sup, [0, c], [lower_s, upper_s])
    #     cdf = np.insert(cdf, [0, c], [cdf[0]/2, cdf[c-1] + ((1 - cdf[c-1])/2)])
    #
    #     return sup, cdf

    def linearly_interpolate(vals, x_axis, y_axis):
        # for each element of vals, get idxs of the two elements in the support that sandwhich it
        all_diffs = vals[:, np.newaxis] - x_axis  # (n, c)
        pos_diffs = np.where(all_diffs >= 0, all_diffs, np.nan)  # (n, c)

        # need to deal separately with outliers
        below_range = np.all(np.isnan(pos_diffs), axis=1)
        above_range = np.all(~np.isnan(pos_diffs), axis=1)
        in_range = ~np.logical_or(below_range, above_range)

        lesser_idxs = np.nanargmin(pos_diffs[in_range], axis=-1)
        greater_idxs = lesser_idxs + 1

        # use linearly interpolation to evaluate cdf(vals)
        in_range_vals = vals[in_range]
        x_diff = in_range_vals - x_axis[lesser_idxs]
        frac = x_diff / (x_axis[greater_idxs] - x_axis[lesser_idxs])
        lesser_yvals = y_axis[lesser_idxs]
        greater_yvals = y_axis[greater_idxs]

        y = np.zeros_like(vals)
        y[in_range] = lesser_yvals + (greater_yvals - lesser_yvals) * frac
        y[below_range] = np.amin(y_axis)
        y[above_range] = np.amax(y_axis)

        return y

    def cdf_fn(x, inverse=False, batch_size=None):
        n, d = x.shape
        if not batch_size: batch_size = n
        y = np.zeros_like(x)
        for i in range(d):
            xi = x[:, i]  # (n, )
            supi = supports[i]  # (c, )
            cdfi = cdfs[i]  # (c, )
            # supi, cdfi = enlarge_domains(xi, supi, cdfi)

            if inverse:
                fn = lambda xi: linearly_interpolate(xi, x_axis=cdfi, y_axis=supi)
            else:
                fn = lambda xi: linearly_interpolate(xi, x_axis=supi, y_axis=cdfi)

            y[:, i] = batched_operation(xi, fn, batch_size)

        return y  # (n, d)

    return cdf_fn


def eval_kdes(kdes, data, log=True, batch_size=None):
    n, d = data.shape
    if not batch_size: batch_size = n

    vals = np.zeros_like(data)
    for i in range(d):
        kde, x = kdes[i], data[:, i]
        val = batched_operation(x, kde.evaluate, batch_size)
        if log: val = np.log(val + 1e-7)
        vals[:, i] = val

    return vals  # (n, d)


def load_uniformized_marginals(dataset_name, which_set="train"):
    dir = os.path.join(density_data_root, dataset_name, "kde_of_marginals/")
    loaded = np.load(os.path.join(dir, "uniformized_data.npz"))
    x = loaded[which_set]
    ldj_per_x = loaded[which_set + "_ldj_per_x"]

    return x, ldj_per_x


def load_kdes_and_cdfs(dataset_name):
    dir = os.path.join(density_data_root, dataset_name, "kde_of_marginals/")
    with open(os.path.join(dir, "kdes"), 'rb') as f:
        marginal_kdes = dill.load(f)

    loaded = np.load(os.path.join(dir, "cdfs.npz"))
    cdf_fn = make_cdf_fn(loaded["supports"], loaded["cdfs"])

    return marginal_kdes, cdf_fn


def plot_deuniformized_data(x, dataset_name, dir_name, layout=None, n_pages=1, name=""):
    y = deuniformize_data(dataset_name, x)
    disp_imdata(y, dataset_name, layout=layout, num_pages=n_pages, dir_name=dir_name, name=name)


def deuniformize_data(dataset_name, x):
    _, cdf_fn = load_kdes_and_cdfs(dataset_name)
    x = x.reshape(x.shape[0], -1)
    y = cdf_fn(x, inverse=True, batch_size=1000)
    return y


def crop_img(x, img_shape, crop_n):
    return x.reshape(-1, *img_shape)[:, crop_n:-crop_n, crop_n:-crop_n, :]


def show_pixel_histograms(data, pixel=None, alpha=1.0, ax=None, show=True):
    """
    Shows the histogram of pixel values, or of a specific pixel if given.
    """
    imgsize = data.shape[1:]

    if pixel is None:
        data = data.flatten()

    else:
        row, col = pixel
        idx = row * imgsize[0] + col
        data = data.reshape(len(data), -1)[:, idx]

    n_bins = int(np.sqrt(len(data)))
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.hist(data, n_bins, density=True, alpha=alpha)
    if show:
        plt.show()


def show_pixel_histograms_rgb(data, pixel=None, alpha=1.0, axs=None, show=True):
    """
    Shows the histogram of pixel values, or of a specific pixel if given.
    """
    img_size = data.shape[1:]
    if pixel is None:
        data_r = data[..., 0].flatten()
        data_g = data[..., 1].flatten()
        data_b = data[..., 2].flatten()

    else:
        row, col = pixel
        idx = row * img_size[0] + col
        data_r = data[..., 0].reshape(len(data), -1)[:, idx]
        data_g = data[..., 1].reshape(len(data), -1)[:, idx]
        data_b = data[..., 2].reshape(len(data), -1)[:, idx]

    n_bins = int(np.sqrt(len(data)))
    if axs is None:
        fig, axs = plt.subplots(3, 1)
    for ax, d, t in zip(axs, [data_r, data_g, data_b], ['r', 'g', 'b']):
        ax.hist(d, n_bins, density=True, alpha=alpha)
        ax.set_title(t + "_pixel_{}".format(pixel))
    if show:
        plt.show()


def show_multiple_pixel_histograms(dset, pixels, dir_name, filename, plot_normal_histogram=True, rgb=True):
    if plot_normal_histogram:
        mnorm = np.random.multivariate_normal(np.zeros(dset.n_dims), dset.cov_mat, dset.N).reshape(dset.N, *dset.img_shape)

    nrow = 3 if rgb else 1
    fig, axs = plt.subplots(nrow, len(pixels))
    for i, pixel in enumerate(pixels):
        if rgb:
            show_pixel_histograms_rgb(dset.x, pixel, axs=axs[:, i], alpha=0.5, show=False)
        else:
            show_pixel_histograms(dset.x, pixel, ax=axs[i], alpha=0.5, show=False)

        if plot_normal_histogram:
            if rgb:
                show_pixel_histograms_rgb(mnorm, pixel, axs=axs[:, i], alpha=0.5, show=False)
            else:
                show_pixel_histograms(mnorm, pixel, ax=axs[i], alpha=0.5, show=False)

    save_fig(dir_name, filename)


def load_percent_excluded_data(trn, val, tst, dataset_name, percent_excluded, flow_type):

    results = []
    for which_set, data in zip(["train", "val", "test"], [trn, val, tst]):
        loaded = np.load(path_join(density_data_root,
                                   dataset_name,
                                   which_set,
                                   "{}_sort_idxs.npz".format(flow_type)
                                   )
                         )
        try:
            x, y = data
        except:
            x, y = data, None

        sort_idxs = loaded["sort_idxs"]
        sorted_x = x[sort_idxs]

        n_data = len(x)
        start_idx = int(n_data*(percent_excluded/100))
        subset_x = sorted_x[start_idx:]
        print("N datapoints after {}% excluded : {}".format(percent_excluded, len(subset_x)))

        if y is not None:
            sorted_y = y[sort_idxs]
            subset_y = sorted_y[start_idx:]
            results.append([subset_x, subset_y])
        else:
            results.append(subset_x)

    return results
