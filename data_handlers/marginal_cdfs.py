import json
import dill
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from __init__ import density_data_root
from data_handlers import data_utils as dutils
from scipy.stats import norm

from utils.misc_utils import *
from utils.experiment_utils import *
from utils.plot_utils import *


def fit_kdes(n_dims, data_save_dir, train_data, val_data):

    bw_multipliers = np.array([0.1*i for i in range(1, 21)])

    mult_to_val_score = [{str(bw): np.inf for bw in bw_multipliers} for _ in range(n_dims)]

    # for each dim, track [best_val_loglik, best_bw_mult, best_kde]
    best_kdes_dict = {"dim_{}".format(i): (-np.inf, -np.inf, None) for i in range(n_dims)}
    for mult in bw_multipliers:
        print("starting kde estimation for bandwidth multiplier: {}".format(mult))

        kdes = dutils.kde_of_marginals(train_data, bw_mult=mult, plot=False)
        kde_vals = dutils.eval_kdes(kdes, val_data, log=True)  # (n, d)
        kde_vals = np.mean(kde_vals, axis=0)  # (d, )

        for i in range(n_dims):
            cur_val = best_kdes_dict["dim_{}".format(i)][0]
            if kde_vals[i] > cur_val:
                best_kdes_dict["dim_{}".format(i)] = [kde_vals[i], mult, kdes[i]]

            mult_to_val_score[i][str(mult)] = kde_vals[i]

    # for each dim, plot each bandwidth against corresponding validation score
    fig, axs = plt.subplots(int(np.ceil(n_dims/2)), 2)
    axs = axs.ravel()
    for i in range(n_dims):
        ax = axs[i]
        val_scores = [mult_to_val_score[i][str(m)] for m in bw_multipliers]
        ax.scatter(bw_multipliers, val_scores)
    save_fig(data_save_dir, "bandwidths_vs_val_scores")

    # save the MoG centres & best bandwidths to disk
    best_bws = [val[1] for val in best_kdes_dict.values()]
    print("best bandwidths: {}".format(best_bws))
    np.savez(os.path.join(data_save_dir, "kde_locs_and_scales"), locs=train_data.T, scales=best_bws)

    best_kdes = [val[2] for val in best_kdes_dict.values()]

    return best_kdes


def plot_kdes(best_kdes, data_save_dir, train_data, val_data):
    dutils.plot_all_marginals(train_data, best_kdes, data_save_dir, "train_best_kdes_of_marginals")
    dutils.plot_all_marginals(val_data, best_kdes, data_save_dir, "val_best_kdes_of_marginals")
    with open(os.path.join(data_save_dir, "kdes"), 'wb') as f:
        dill.dump(best_kdes, f)


def compute_cdf(best_kdes, data_save_dir, train_dp, val_dp, test_dp, dataset_name):
    print("computing the cdf...")
    # calculate the cdf of the best kdes over a grid, and save these values to disk
    supports = np.array([kde.support for kde in best_kdes])
    cdfs = np.array([kde.cdf for kde in best_kdes])
    np.savez(os.path.join(data_save_dir, "cdfs"), supports=supports, cdfs=cdfs)
    # loaded = np.load(os.path.join(data_save_dir, "cdfs.npz"))
    # supports, cdfs = loaded["supports"], loaded["cdfs"]
    print("finished computing the cdf")

    print("applying cdf transform to the data...")
    # plot cdf(data), which should be approximately uniform
    cdf_fn = dutils.make_cdf_fn(supports, cdfs)
    uniformized_trn_data, trn_ldj_per_x = uniformize_data(cdf_fn, best_kdes, train_dp.data)
    uniformized_val_data, val_ldj_per_x = uniformize_data(cdf_fn, best_kdes, val_dp.data)
    # if "2d" in dataset_name:
    #     uniformized_tst_data, tst_ldj_per_x = uniformize_data(cdf_fn, best_kdes, train_dp.data_source.tst.x)
    # else:
    #     uniformized_tst_data, tst_ldj_per_x = uniformize_data(cdf_fn, best_kdes, test_dp.data)
    uniformized_tst_data, tst_ldj_per_x = uniformize_data(cdf_fn, best_kdes, test_dp.data)

    np.savez(os.path.join(data_save_dir, "uniformized_data"),
             train=uniformized_trn_data,
             train_ldj_per_x=trn_ldj_per_x+train_dp.ldj,
             val=uniformized_val_data,
             val_ldj_per_x=val_ldj_per_x+val_dp.ldj,
             test=uniformized_tst_data,
             test_ldj_per_x=tst_ldj_per_x+test_dp.ldj)

    print("plotting the uniformized data...")
    dutils.plot_all_marginals(uniformized_trn_data, save_dir=data_save_dir, filename="train_uniformized_marginals")
    dutils.plot_all_marginals(uniformized_val_data, save_dir=data_save_dir, filename="val_uniformized_marginals")


def uniformize_data(cdf_fn, kdes, data):
    ldj_per_x = np.sum(dutils.eval_kdes(kdes, data, log=True, batch_size=1000), axis=-1)
    uniformized_data = cdf_fn(data, batch_size=1000)
    return uniformized_data, ldj_per_x


def plot_densities_using_mog(best_kdes, data_save_dir, train_data):
    fig, axs = plt.subplots(10, 2)
    axs = axs.ravel()
    for i in range(20):
        kde_i = best_kdes[i]
        gaussians = [norm(loc=loc, scale=kde_i.bw) for loc in train_data[:, i]]

        test_grid = kde_i.support
        densities = np.mean(np.array([g.pdf(test_grid) for g in gaussians]), axis=0)

        axs[i].plot(test_grid, densities)
    save_fig(data_save_dir, "mog_train_densities_0")


# noinspection PyUnresolvedReferences
def main():
    """fit KDE to marginals of data, choosing the bandwidth via cross-validation"""
    np.set_printoptions(precision=3)

    parser = ArgumentParser(description='Uniformize marginals of a dataset', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_path', type=str, default="mnist/model/1")
    parser.add_argument('--num_mixtures', type=int, default=1024)
    parser.add_argument('--load_kde_from_disk', type=int, default=-1)
    args = parser.parse_args()
    with open(project_root + "configs/{}.json".format(args.config_path)) as f:
        config = json.load(f)
    config = merge_dicts(*list(config.values()))  # json is 2-layers deep, flatten it
    dataset_name = config["dataset_name"]
    globals().update(config)

    if (data_args is not None) and ("img_shape" in data_args):
        data_args["img_shape"] = [np.prod(np.array(data_args["img_shape"]))]  # we want flattened images
    if "2d" in dataset_name: dataset_name = "2d/{}".format(dataset_name)

    data_save_dir = os.path.join(density_data_root, dataset_name, "kde_of_marginals/")
    os.makedirs(data_save_dir, exist_ok=True)

    train_dp, val_dp, test_dp = load_data_providers_and_update_conf(config, include_test=True)

    n_samples, n_dims = train_dp.data.shape
    train_data = train_dp.data[:min(n_samples, args.num_mixtures)]  # use at most num_mixtures for kde estimate (for speed)

    if args.load_kde_from_disk == -1:
        # fit a non-parameteric KDE to each dimension of the data (using cross-validation to select bandwidth)
        bw_path = os.path.join(data_save_dir, "locs_and_scales.npz")
        if os.path.isfile(bw_path):
            print("loading pre-computed bandwidths from disk...")
            bws = np.load(bw_path)["scales"]
            best_kdes = dutils.kde_of_marginals(train_data, bw_mult=bws, plot=False)
        else:
            print("estimating bandwidths via cross-validation...")
            best_kdes = fit_kdes(n_dims, data_save_dir, train_data, val_dp.data)
    else:
        dir = os.path.join(density_data_root, dataset_name, "kde_of_marginals/")
        with open(os.path.join(dir, "kdes"), 'rb') as f:
            best_kdes = dill.load(f)

    # plot_densities_using_mog(best_kdes, data_save_dir, train_data)

    # plot best kdes against histograms and samples
    # plot_kdes(best_kdes, data_save_dir, train_data, val_dp.data)

    # compute cdf, apply it to the data, and save the results
    # compute_cdf(best_kdes, data_save_dir, train_dp, val_dp, test_dp, dataset_name)


if __name__ == "__main__":
    main()
