
import json
import numpy as np
import os
from time import gmtime, strftime

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy
from itertools import product
from __init__ import project_root
from shutil import rmtree


def make_base_config():
    data_config = {
        "dataset_name": "",
        "data_seed": 464355,
        # dict of args passed to make_density_data_providers() in data_providers.py
        "data_args": {},
        "frac": 1,  # fraction of data to use (useful for fast testing)
        # value of nu in the NCE (logistic) objective function (note: this doesn't alter size of minibatch)
        "objective_nu": 1,

        "do_mutual_info_estimation": False,
        "noise_dist_name": None,
        "data_dist_name": None,  # for toy problems where data_dist is known

        "waymark_mechanism": "linear_combinations",  # either 'linear_combinations' or 'dimwise_mixing'
        "shuffle_waymarks": False,  # if shuffle_waymarks=True, consecutive waymark samples will no longer be coupled
        "initial_waymark_indices": [0, 1],
        "linear_combo_alphas": [0.0, 1.0],
        "create_waymarks_in_zspace": False,
        "dimwise_mixing_strategy": "fixed_single_order",
        "n_event_dims_to_mix": None,
        # if waymark_method == 'dimwise_mixing', this is the number of new dimensions we mix with each successive waymark
        "waymark_mixing_increment": 1,
    }
    optim_config = {
        "loss_function": "logistic",
        "energy_restore_path": None,
        "optimizer": "adam",
        "n_epochs": 300,
        "n_batch": 256,
        "energy_lr": 1e-4,
        "scale_param_lr_multiplier": 10.0,
        "energy_reg_coef": 0.,

        # Can apply different dropout rates to each bridge, by specifying an `start_rate' for the first bridge,
        # an 'end_rate' for the final bridge, and a 'power' that controls the interpolation between the start & end rates.
        # by default, dropout is turned off (and isn't used for final results in the paper)
        "dropout_params": [False, 0.0, 0.0, 2.0],  # (include_final_layer, start_rate, end_rate, power)
        "max_spectral_norm_params": None,  # 'None' means don't use spectral_norm, which is default
        "just_track_spectral_norm": False,
        "label_smoothing_alpha": 0.0,  # 0.0 means no label smoothing (which is the default)
        "one_sided_smoothing": True,
        "loss_decay_factor": 1.0,  # 1.0 means no reweighting of the loss (which is the default)
        "patience": np.inf,  # terminate learning if val loss doesn't improve after this many epochs
        "save_every_x_epochs": None  # may be set to an integer x. Otherwise, just save best model so far
    }
    architecture_config = {
        "network_type": "mlp",
        "mlp_hidden_size": 128,
        "mlp_output_size": None,   # defaults to mlp_hidden_size
        "mlp_n_blocks": 1,
        "activation_name": "leaky_relu",
        "use_residual_mlp": True,
        "use_fc_layer": True,  # use fully connected layer at end of convnet
        "final_pool_shape": (2, 2),
        "conv_kernel_shape": (3, 3),
        "use_global_sum_pooling": True,
        "use_attention": True,
        "head_type": "quadratic",
        "use_single_head": False,
        "use_cond_scale_shift": True,  # bridge-conditional scale+shift per hidden unit
        "shift_scale_per_channel": False,  # instead of per hidden unit, make the bridge parameters channel-wise
        "use_instance_norm": False,
    }
    ais_config = {
        "ais_n_chains": 100,
        "ais_total_n_steps": 10000,
        "only_sample_n_chains": 100,
        "only_sample_total_n_steps": 1000,
        "ais_n_leapfrog_steps": 10
    }
    flow_config = {
        "flow_lr": 5e-4,
        "flow_keep_prob": 1.0,
        "flow_n_bijectors": 5,  # doesn't apply to glow - see 'glow_depth'
        "flow_num_layers_or_blocks": 2,  # doesn't apply to glow
        "flow_hidden_size": 256,
        "flow_activation": "relu",
        "flow_type": "GaussianCopula",
        "glow_depth": 8,
        "glow_use_split": False,
        "glow_coupling_type": "rational_quadratic",
        "flow_num_spline_bins": 8,
        "glow_temperature": 1.0,
        "mogmade_n_mixture_components": 10,
        "num_splines": 1,  # for gauss copula
        "spline_interval_min": -3,  # for gauss copula
        "nbins_for_splines": 128,  # for gauss copula
        "logit_copula_marginals": False  # for gauss copula
    }

    config = {
        "data": data_config,
        "architecture": architecture_config,
        "optimisation": optim_config,
        "ais": ais_config,
        "flow": flow_config
    }

    return config


def make_1d_configs():
    config = make_base_config()
    config["data"]["n_dims"] = 1

    config["architecture"]["mlp_hidden_size"] = 32
    config["architecture"]["mlp_n_blocks"] = 2

    config["optimisation"]["n_batch"] = 512
    config["optimisation"]["n_epochs"] = 250

    return config


# def make_1d_gauss_configs():
#     config = make_1d_configs()
#     config["data"]["dataset_name"] = "1d_gauss"
#     config["data"]["data_args"] = {"n_gaussians": 1, "mean": 0, "std": 1e-6, "n_samples": 10000, "outliers": False}
#     config["data"]["noise_dist_name"] = "gaussian"
#     config["data"]["noise_dist_gaussian_loc"] = 0.0
#     config["data"]["noise_dist_gaussian_std"] = 1.0
#
#     config["architecture"]["network_type"] = "quadratic"
#     config["architecture"]["quadratic_head_use_linear_term"] = True
#     config["optimisation"]["energy_lr"] = 1e-2
#     config["optimisation"]["n_batch"] = 1000
#
#     dargs1 = {"n_gaussians": 1, "mean": 0.0, "std": 1e-6, "n_samples": 10000}
#     dargs2 = {"n_gaussians": 1, "mean": -1.0, "std": 0.08, "n_samples": 10000}
#     dargs3 = {"n_gaussians": 1, "mean": -2.0, "std": 0.08, "n_samples": 10000}
#     dargs4 = {"n_gaussians": 1, "mean": -5.0, "std": 1.0, "n_samples": 10000}
#
#     p1 = [["data", "data", "data", "data", "data"],
#           ["data_args",
#            "noise_dist_gaussian_loc",
#            "noise_dist_gaussian_std",
#            "linear_combo_alphas",
#            "initial_waymark_indices"],
#           [
#               [dargs1, 0.0, 1.0, *get_poly_wmark_coefs(num=4, p=5.0)],
#               [dargs2, 2.0, 0.15, *get_poly_wmark_coefs(num=20, p=1.0)],
#               [dargs3, 2.0, 0.15, *get_poly_wmark_coefs(num=30, p=1.0)],
#               [dargs4, 5.0, 1.0, *get_poly_wmark_coefs(num=10, p=1.0)],
#           ]
#           ]
#
#     generate_configs_for_gridsearch(config, "model", p1)


def make_1d_gauss_configs():
    config = make_1d_configs()
    config["data"]["dataset_name"] = "1d_gauss"
    config["data"]["data_args"] = {"n_gaussians": 1, "mean": 0, "std": 1e-6, "n_samples": 10000, "outliers": False}

    config["data"]["noise_dist_gaussian_loc"] = 0.0
    config["data"]["noise_dist_gaussian_std"] = 1.0

    # config["architecture"]["network_type"] = "linear"
    config["architecture"]["network_type"] = "quadratic"
    config["architecture"]["quadratic_head_use_linear_term"] = False
    config["optimisation"]["energy_lr"] = 1e-3
    config["optimisation"]["n_batch"] = 1000

    p1 = ["optimisation", "loss_function", ["logistic", "nwj", "lsq"]]
    p2 = [["data", "data", "data"],
          ["noise_dist_name", "linear_combo_alphas", "initial_waymark_indices"],
          [
              ["gaussian", *get_poly_wmark_coefs(num=2, p=1.0)],
              ["gaussian", *get_poly_wmark_coefs(num=5, p=7.0)],
          ]
          ]

    generate_configs_for_gridsearch(config, "model", p1, p2)

#
# def make_gaussians_configs():
#     config = make_base_config()
#     config["data"]["dataset_name"] = "gaussians"
#     config["data"]["data_dist_name"] = "gaussian"
#     config["data"]["noise_dist_name"] = "gaussian"
#
#     config["optimisation"]["n_epochs"] = 250
#     config["optimisation"]["n_batch"] = 512
#     config["optimisation"]["patience"] = 50
#     config["optimisation"]["save_every_x_epochs"] = 10
#
#     # config["architecture"]["network_type"] = "mlp"
#     config["architecture"]["network_type"] = "quadratic"
#     config["architecture"]["quadratic_constraint_type"] = "symmetric_pos_diag"
#     config["architecture"]["quadratic_head_use_linear_term"] = True
#
#     config["ais"]["ais_n_chains"] = 1000
#     config["ais"]["ais_total_n_steps"] = 1000
#
#     data_args1 = {"n_samples": 100000, "n_dims": 40, "mean": -1.0, "std": 1.0}
#     data_args2 = {"n_samples": 100000, "n_dims": 160, "mean": -0.5, "std": 1.0}
#     data_args3 = {"n_samples": 100000, "n_dims": 320, "mean": -0.5, "std": 1.0}
#
#     p1 = [["data", "data", "data", "data", "data", "data", "optimisation"],
#           ["linear_combo_alphas", "initial_waymark_indices", "n_dims",
#            "data_args", "noise_dist_gaussian_loc", "noise_dist_gaussian_std", "energy_lr"],
#           [
#               [*get_poly_wmark_coefs(num=9, p=1.0), data_args1["n_dims"], data_args1, 1.0, 1.0, 1e-4],
#               [*get_poly_wmark_coefs(num=2, p=1.0), data_args1["n_dims"], data_args1, 1.0, 1.0, 5e-4],
#
#               [*get_poly_wmark_coefs(num=13, p=1.0), data_args2["n_dims"], data_args2, 0.6, 1.0, 1e-4],
#               [*get_poly_wmark_coefs(num=2, p=1.0), data_args2["n_dims"], data_args2, 0.6, 1.0, 5e-4],
#
#               [*get_poly_wmark_coefs(num=17, p=1.0), data_args3["n_dims"], data_args3, 0.5, 1.0, 1e-4],
#               [*get_poly_wmark_coefs(num=2, p=1.0), data_args3["n_dims"], data_args3, 0.5, 1.0, 5e-4],
#           ]
#           ]
#
#     generate_configs_for_gridsearch(config, "model", p1)


def make_gaussians_configs():
    config = make_base_config()
    config["data"]["dataset_name"] = "gaussians"
    config["data"]["n_dims"] = 80
    config["data"]["data_args"] = {"n_samples": 100000, "dims": config["data"]["n_dims"], "true_mutual_info": 20}
    config["data"]["data_dist_name"] = "gaussian"

    config["data"]["noise_dist_name"] = "gaussian"

    config["optimisation"]["n_epochs"] = 250
    config["optimisation"]["n_batch"] = 512
    config["optimisation"]["patience"] = 50
    config["optimisation"]["save_every_x_epochs"] = 10

    # config["architecture"]["network_type"] = "mlp"
    config["architecture"]["network_type"] = "quadratic"
    config["architecture"]["quadratic_constraint_type"] = "symmetric_pos_diag"
    config["architecture"]["quadratic_head_use_linear_term"] = True

    config["ais"]["ais_n_chains"] = 1000
    config["ais"]["ais_total_n_steps"] = 1000

    data_args1 = {"n_samples": 100000, "n_dims": 40, "true_mutual_info": 10}
    data_args2 = {"n_samples": 100000, "n_dims": 80, "true_mutual_info": 20}
    data_args3 = {"n_samples": 100000, "n_dims": 160, "true_mutual_info": 40}
    data_args4 = {"n_samples": 100000, "n_dims": 320, "true_mutual_info": 80}

    p1 = [["data", "data", "data", "data", "optimisation"],
          ["linear_combo_alphas", "initial_waymark_indices", "n_dims", "data_args", "energy_lr"],
          [
              [*get_poly_wmark_coefs(num=3, p=1.0), data_args1["n_dims"], data_args1, 1e-4],
              [*get_poly_wmark_coefs(num=2, p=1.0), data_args1["n_dims"], data_args1, 5e-4],

              [*get_poly_wmark_coefs(num=5, p=1.0), data_args2["n_dims"], data_args2, 1e-4],
              [*get_poly_wmark_coefs(num=2, p=1.0), data_args2["n_dims"], data_args2, 5e-4],

              [*get_poly_wmark_coefs(num=7, p=1.0), data_args3["n_dims"], data_args3, 1e-4],
              [*get_poly_wmark_coefs(num=2, p=1.0), data_args3["n_dims"], data_args3, 5e-4],

              [*get_poly_wmark_coefs(num=9, p=1.0), data_args4["n_dims"], data_args4, 1e-4],
              [*get_poly_wmark_coefs(num=2, p=1.0), data_args4["n_dims"], data_args4, 5e-4],
          ]
          ]

    generate_configs_for_gridsearch(config, "model", p1)


def make_mnist_configs():
    config = make_base_config()
    config["data"]["dataset_name"] = "mnist"
    config["data"]["n_dims"] = 784
    config["data"]["data_args"] = {"dequantize": True, "logit": True, "img_shape": [28, 28, 1]}
    config["data"]["create_waymarks_in_zspace"] = True

    # First, we need to train a copula/flow model. This model will be saved under a specific time identifier
    # (e.g. 20200408-1721_0), and then, to train a corresponding TRE model, we need to set config["flow]["flow_id"]
    _make_mnist_noise_dist_config(config, noise_type="GaussianCopula")
    # _make_mnist_noise_dist_config(config, noise_type="GLOW")

    config["architecture"]["network_type"] = "resnet"
    config["architecture"]["channel_widths"] = [[64], [64, 64], [64, 64], [128, 128], [128, 128]]
    config["architecture"]["mlp_hidden_size"] = 128
    config["optimisation"]["n_epochs"] = 150

    p1 = [["data", "data", "data", "flow", "flow", "optimisation"],
          ["linear_combo_alphas", "initial_waymark_indices", "noise_dist_name", "flow_type", "flow_id", "n_batch"],
          [
              [*get_poly_wmark_coefs(num=11, p=1.0, drop_first=True), "full_covariance_gaussian", None, None, 250],
              [*get_poly_wmark_coefs(num=16, p=1.0, drop_first=True), "full_covariance_gaussian", None, None, 375],
              [*get_poly_wmark_coefs(num=21, p=1.0, drop_first=True), "full_covariance_gaussian", None, None, 500],
              [*get_poly_wmark_coefs(num=26, p=1.0, drop_first=True), "full_covariance_gaussian", None, None, 625],
              [*get_poly_wmark_coefs(num=31, p=1.0, drop_first=True), "full_covariance_gaussian", None, None, 750],

              [*get_poly_wmark_coefs(num=11, p=1.0, drop_first=True), "flow", "GaussianCopula", "20200408-1721_0", 250],
              [*get_poly_wmark_coefs(num=16, p=1.0, drop_first=True), "flow", "GaussianCopula", "20200408-1721_0", 375],
              [*get_poly_wmark_coefs(num=21, p=1.0, drop_first=True), "flow", "GaussianCopula", "20200408-1721_0", 500],
              [*get_poly_wmark_coefs(num=26, p=1.0, drop_first=True), "flow", "GaussianCopula", "20200408-1721_0", 625],

              [*get_poly_wmark_coefs(num=11, p=1.0, drop_first=True), "flow", "GLOW", "20200220-1137_2", 125],
          ]
          ]
    # p2 = ["data", "create_waymarks_in_zspace", [True, False]]

    generate_configs_for_gridsearch(config, "model", p1)


def _make_mnist_noise_dist_config(config, noise_type):

    config["flow"]["flow_hidden_size"] = 64
    config["flow"]["flow_num_spline_bins"] = 8

    if noise_type == "full_covariance_gaussian":
        config["data"]["noise_dist_name"] = "full_covariance_gaussian"

    elif noise_type == "GaussianCopula":
        config["data"]["noise_dist_name"] = "flow"
        config["flow"]["flow_type"] = "GaussianCopula"
        config["data"]["flow_id"] = "20200408-1721_0"

        config["optimisation"]["n_epochs"] = 400
        config["optimisation"]["n_batch"] = 512
        config["optimisation"]["patience"] = 200
        p1 = ["flow", "flow_lr", [1e-4]]
        generate_configs_for_gridsearch(config, "flow", p1)

    elif noise_type == "GLOW":
        config["data"]["noise_dist_name"] = "flow"
        config["flow"]["flow_type"] = "GLOW"
        config["data"]["flow_id"] = "20200220-1137_2"

        config["flow"]["flow_keep_prob"] = 0.9
        config["optimisation"]["n_epochs"] = 1024
        config["optimisation"]["n_batch"] = 256
        config["optimisation"]["patience"] = 1500

        p1 = ["flow", "flow_keep_prob", [0.9]]
        generate_configs_for_gridsearch(config, "flow", p1)
    else:
        raise NotImplementedError


def make_multiomniglot_configs():
    config = make_base_config()
    config["data"]["dataset_name"] = "multiomniglot"

    config["data"]["do_mutual_info_estimation"] = True
    config["data"]["waymark_mechanism"] = "dimwise_mixing"
    config["data"]["dimwise_mixing_strategy"] = "fixed_single_order"
    config["data"]["n_event_dims_to_mix"] = 1  # don't mix pixel values, just image blocks

    config["architecture"]["use_attention"] = False
    config["architecture"]["use_average_pooling"] = False
    config["optimisation"]["n_batch"] = 256

    stacked = False  # if False, use spatial layout of images

    p1 = [["data", "data", "data", "data",
           "architecture", "architecture", "architecture", "architecture", "architecture", "optimisation"],
          ["data_args", "n_dims", "waymark_mixing_increment", "initial_waymark_indices",
           "network_type", "channel_widths", "init_kernel_shape", "init_kernel_strides", "mlp_hidden_size", "n_epochs"],
          [
              # 1 image, 1 ratio, various widths
              _get_single_multiomniglot_hparam_setting(
                  n_imgs=1, stacked=stacked, mixing_increment=1, network_type="resnet", base_channel_width=32,
                  init_kernel_shape=(5, 5), init_kernel_strides=(3, 3), base_mlp_width=300),
              _get_single_multiomniglot_hparam_setting(
                  n_imgs=1, stacked=stacked, mixing_increment=1, network_type="resnet", base_channel_width=32,
                  init_kernel_shape=(5, 5), init_kernel_strides=(3, 3), base_mlp_width=500),

             # 4 imgs, 4 ratios, various widths
              _get_single_multiomniglot_hparam_setting(
                  n_imgs=4, stacked=stacked, mixing_increment=1, network_type="resnet", base_channel_width=32,
                  init_kernel_shape=(5, 5), init_kernel_strides=(3, 3), base_mlp_width=300),
              _get_single_multiomniglot_hparam_setting(
                  n_imgs=4, stacked=stacked, mixing_increment=1, network_type="resnet", base_channel_width=32,
                  init_kernel_shape=(5, 5), init_kernel_strides=(3, 3), base_mlp_width=500),

              # 4 imgs, 1 ratio, various widths
              _get_single_multiomniglot_hparam_setting(
                  n_imgs=4, stacked=stacked, mixing_increment=4, network_type="resnet", base_channel_width=32,
                  init_kernel_shape=(5, 5), init_kernel_strides=(3, 3), base_mlp_width=150),
              _get_single_multiomniglot_hparam_setting(
                  n_imgs=4, stacked=stacked, mixing_increment=4, network_type="resnet", base_channel_width=32,
                  init_kernel_shape=(5, 5), init_kernel_strides=(3, 3), base_mlp_width=300),
              _get_single_multiomniglot_hparam_setting(
                  n_imgs=4, stacked=stacked, mixing_increment=4, network_type="resnet", base_channel_width=32,
                  init_kernel_shape=(5, 5), init_kernel_strides=(3, 3), base_mlp_width=500),

              # 9 imgs, 9 ratios, various widths
              _get_single_multiomniglot_hparam_setting(
                  n_imgs=9, stacked=stacked, mixing_increment=1, network_type="resnet", base_channel_width=32,
                  init_kernel_shape=(5, 5), init_kernel_strides=(3, 3), base_mlp_width=300),
              _get_single_multiomniglot_hparam_setting(
                  n_imgs=9, stacked=stacked, mixing_increment=1, network_type="resnet", base_channel_width=32,
                  init_kernel_shape=(5, 5), init_kernel_strides=(3, 3), base_mlp_width=500),

              # 9 imgs, 1 ratio, various widths
              _get_single_multiomniglot_hparam_setting(
                  n_imgs=9, stacked=stacked, mixing_increment=9, network_type="resnet", base_channel_width=32,
                  init_kernel_shape=(5, 5), init_kernel_strides=(3, 3), base_mlp_width=150),
              _get_single_multiomniglot_hparam_setting(
                  n_imgs=9, stacked=stacked, mixing_increment=9, network_type="resnet", base_channel_width=32,
                  init_kernel_shape=(5, 5), init_kernel_strides=(3, 3), base_mlp_width=300),
              _get_single_multiomniglot_hparam_setting(
                  n_imgs=9, stacked=stacked, mixing_increment=9, network_type="resnet", base_channel_width=32,
                  init_kernel_shape=(5, 5), init_kernel_strides=(3, 3), base_mlp_width=500),
           ]
          ]

    generate_configs_for_gridsearch(config, "model", p1)


def _get_single_multiomniglot_hparam_setting(n_imgs,
                                             stacked,
                                             mixing_increment,
                                             network_type,
                                             base_channel_width=32,
                                             init_kernel_shape=(3, 3),
                                             init_kernel_strides=(1, 1),
                                             base_mlp_width=128,
                                             ):
    n_sqrt = int(n_imgs**0.5)
    d_args = _get_multiomniglot_data_args(n_imgs, stacked)
    n_dims = int(np.prod(d_args["img_shape"]))
    wmark_idxs = list(range(int(n_imgs/mixing_increment) + 1))
    channel_widths = [[base_channel_width], [base_channel_width]*2, [2*base_channel_width]*2, [2*base_channel_width]*2]
    mlp_width = base_mlp_width*n_sqrt
    n_epochs = int((2000 * mixing_increment) / n_imgs)

    hparams = [
        d_args, n_dims, mixing_increment, wmark_idxs, network_type,
        channel_widths, init_kernel_shape, init_kernel_strides, mlp_width, n_epochs
    ]

    return hparams


def _get_multiomniglot_data_args(n_imgs, stacked):
    n_sqrt = int(n_imgs**0.5)
    if stacked:
        return {"n_imgs": n_imgs, "stacked": True, "img_shape": [28, 28, n_imgs, 2]}
    else:
        return {"n_imgs": n_imgs, "stacked": False, "img_shape": [n_sqrt*28, n_sqrt*28, 1, 2]}  # spatially arranged


def generate_configs_for_gridsearch(config, name, *args):
    keys1 = [arg[0] for arg in args]
    keys2 = [arg[1] for arg in args]
    params = [arg[2] for arg in args]

    configs = []
    params_grid = product(*params)
    for i, p in enumerate(params_grid):
        config_i = deepcopy(config)
        for key1, key2, val in zip(keys1, keys2, p):
            if not isinstance(key1, list):
                key1, key2, val = [key1], [key2], [val]
            assert len(key1) == len(key2) == len(val), "the *args input to this function contains " \
                "elements of the form [keys1, keys2, associated_vals_to_grid_search_over]. \n If keys1 " \
                "is a list (e.g ['architecture', 'optimisation]), then key2 must also be a list of " \
                "same length (e.g. ['num_layers', 'energy_lr']) and the associated vals should be of the form " \
                "[[2, '1e-3] [2, '1e-4], [3, '1e-4]]"

            for subkey1, subkey2, subval in zip(key1, key2, val):
                config_i[subkey1][subkey2] = subval

        save_config(config_i, name, i)
        configs.append(config_i)

    # save the gridsearch hparams to a .txt file
    if name == "model":
        hparams_dir = os.path.join(project_root, "saved_models", config["data"]["dataset_name"], "hparams")
        os.makedirs(hparams_dir, exist_ok=True)
        hparams_to_txt_file(os.path.join(hparams_dir, "{}_hparams.txt".format(time_id)), args)

    return configs

def hparams_to_txt_file(filename, hparams):
    with open(filename, 'w+') as f:
        for h in hparams:
            f.write("\n")
            for i, hi in enumerate(h):
                if i <2:
                    f.write(str(hi) + "\n")
                else:
                    for hii in hi:
                        f.write(str(hii) + "\n")


def check_valid_config(c):
    pass


def update_waymark_method_settings(c):
    pass


def update_config(c, i):

    dataset_name = c["data"]["dataset_name"]
    c["data"]["fig_dir_name"] = "{}figs/{}/{}/".format(project_root, dataset_name, time_id)
    c["data"]["save_dir"] = project_root + "{}/{}/{}_{}/".format("saved_models", dataset_name, time_id, i)

    p_exc = c["data"]["data_args"].get("percent_excluded", None)
    if p_exc is not None:
        c["data"]["data_args"]["flow_type"] = c["flow"]["flow_type"]

    c["data"]["init_num_ratios"] = len(c["data"]["initial_waymark_indices"]) - 1
    c["data"]["waymark_idxs"] = c["data"]["initial_waymark_indices"]
    c["data"]["bridge_idxs"] = c["data"]["initial_waymark_indices"][:-1]

    n_ratios = c["data"]["init_num_ratios"]
    c["optimisation"]["num_losses"] = n_ratios

    update_waymark_method_settings(c)


def get_flow_base_mixture_stds_and_weights(num, start_std, end_std, weight_factor=1.0):
    stds = [x for x in np.linspace(start_std, end_std, num=num)]
    weights = np.linspace(1.0, 1.0/weight_factor, num=num)
    weights /= np.sum(weights)
    weights = [w for w in weights]
    return [stds, weights]


def get_poly_wmark_coefs(num, p, mini=None, drop_first=False):
    if mini is not None:
        scales = [0.0] + [(x/(num-1)) ** p for x in np.linspace((num-1) * (mini ** (1 / p)), num-1, num-1)]
    else:
        scales = [(x/num) ** p for x in np.linspace(0.0, num, num)]

    start = 1 if drop_first else 0
    idxs = np.arange(start, len(scales))

    return scales, [int(i) for i in idxs]


def get_symmetric_poly_noise_scales(n, p, mini=None, drop_first=False):
    m = int(n/2)+1
    if mini is not None:
        scales = [0.0] + [(x/(m-1)) ** p for x in np.linspace((m-1) * (mini ** (1 / p)), m-1, m-1)]
    else:
        scales = [(x/m) ** p for x in np.linspace(0.0, m, m)]

    scales = np.array(scales) / 2
    scales = np.concatenate([scales, 1-scales[:-1][::-1]])
    scales = [x for x in scales]

    start = 1 if drop_first else 0
    idxs = np.arange(start, len(scales))

    return scales, [int(i) for i in idxs]


def save_config(config, name, i):
    update_config(config, i)
    check_valid_config(config)
    save_dir = project_root + 'configs/{}/{}/'.format(config["data"]["dataset_name"], name)
    if i == 0:
        rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)
    with open(save_dir + '{}.json'.format(i), 'w') as fp:
        json.dump(config, fp, indent=4)


def main():
    parser = ArgumentParser(description='Create configs for TRE',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--time_id', type=str, help="model config ids", default=None)
    args = parser.parse_args()

    time_id = strftime('%Y%m%d-%H%M', gmtime()) if not args.time_id else args.time_id
    globals().update({"time_id": time_id})

    make_1d_gauss_configs()
    make_gaussians_configs()
    make_mnist_configs()
    make_multiomniglot_configs()


if __name__ == "__main__":
    main()
