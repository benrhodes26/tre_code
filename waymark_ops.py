import numpy as np
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

from experiment_ops import build_blockwise_correlated_gaussian_waymarks, build_noise_dist
from utils.misc_utils import *
from utils.tf_utils import *
from utils.experiment_utils import *
from utils.plot_utils import *


def tf_build_noise_additive_waymarks_on_the_fly(data, waymark_idxs, config, logger, noise_dist=None):
    logger.info("--------------------------------------")
    logger.info("Generating noise-additive waymarks on-the-fly!")
    logger.info("--------------------------------------")

    batch_size_per_waymark, event_shape, event_ones, n_data, n_waymarks, waymark_shp = \
        _tf_get_shapes_for_additive_wmarks(data, waymark_idxs, config, logger)

    if noise_dist is None:
        noise_dist = build_noise_dist(config.noise_dist_name, data, config, event_shape)

    if config.noise_dist_name == "flow" and config.create_waymarks_in_zspace:
        data = tf.reshape(noise_dist.inverse(data), [-1, *event_shape])

    if config.shuffle_waymarks:
        # by default, we do *not* `shuffle_waymarks' i.e. we use coupled samples from waymark trajectories
        waymark_data = tf.reshape(data[:n_data], waymark_shp)  # (batch_size, n_waymarks, *event_shape)
    else:
        waymark_data = tf.tile(tf.expand_dims(data, axis=1), [1, n_waymarks] + event_ones)  # (batch_size, n_waymarks, *event_shape)

    data_multipliers, noise_multipliers = \
        _tf_get_mults_for_additive_wmarks(waymark_idxs, config, event_ones, waymark_shp)

    noise = _tf_get_additive_noise(noise_dist, batch_size_per_waymark, config, event_shape, event_ones, n_waymarks, noise_multipliers)

    waymark_data *= data_multipliers
    waymark_data += noise

    if config.noise_dist_name == "flow" and config.create_waymarks_in_zspace:
        waymark_data = tf.reshape(waymark_data, [-1, n_waymarks, np.prod(event_shape)])
        waymark_data = noise_dist.forward(waymark_data, collapse_wmark_dims=True)

    # return waymark_data  # (batch_size, n_waymarks, *event_shape)
    return AttrDict(locals())


def tf_build_dimwise_mixing_waymarks_on_the_fly(data, waymark_idxs,
                                                dimwise_mixing_ordering, config, logger, noise_dist=None):
    logger.info("--------------------------------------")
    logger.info("Generating dimension-wise-mixing-based waymarks on-the-fly!")
    logger.info("--------------------------------------")

    if config.do_mutual_info_estimation:
        x, y = tf.unstack(data, axis=-1)  # each (2*batch_size, *event_dims)
        positive_samples, negative_samples = tf.split(y, 2, axis=0)  # each (batch_size, *event_dims)
    else:
        positive_samples = data

    _, event_dims, event_ones, n_data, n_waymarks, _ = \
        _tf_get_shapes_for_additive_wmarks(positive_samples, waymark_idxs, config, logger)
    n_event_dims = len(event_dims)

    if not config.do_mutual_info_estimation:
        if noise_dist is None:
            noise_dist = build_noise_dist(config.noise_dist_name, positive_samples, config)
        negative_samples = noise_dist.sample(n_data)  # (batch_size, *event_dims)

    positive_samples, negative_samples = preprocess_samples_for_dimwise_mixing(
        positive_samples, negative_samples, n_event_dims, n_waymarks, event_ones, config)

    map_input = (positive_samples, negative_samples, waymark_idxs)
    def mix_fn(u):
        mask = dimwise_mixing_ordering <= u[2] * config.waymark_mixing_increment  # (batch_size, *event_dims)
        mask = tf.cast(mask, u[0].dtype)
        return ((1 - mask) * u[0]) + (mask * u[1])  # (batch_size, *event_dims)

    waymark_data = tf.map_fn(mix_fn, map_input, dtype=positive_samples.dtype)  # (n_waymarks, batch_size, *event_dims)

    waymark_data = revert_preprocessing_for_dimwise_mixing(waymark_data, n_event_dims, config)

    if config.do_mutual_info_estimation:
        waymark_data = (tf.split(x, 2, axis=0)[0], waymark_data)

    return AttrDict(locals())


def revert_preprocessing_for_dimwise_mixing(waymark_data, n_event_dims, config):

    if config.n_event_dims_to_mix is not None:
        # waymark data has shape (n_waymarks, *extra_event_dims, batchsize, *event_dims_to_mix)
        extra_event_dims_pos = [1 + i for i in range(n_event_dims - config.n_event_dims_to_mix)]
        batch_idx = extra_event_dims_pos[-1] + 1
        event_dims_to_mix_pos = [batch_idx + 1 + i for i in range(config.n_event_dims_to_mix)]
        transpose_idxs = [0, batch_idx, *extra_event_dims_pos, *event_dims_to_mix_pos]
        waymark_data = tf.transpose(waymark_data, transpose_idxs)  # (n_waymark, batchsize, *event_dims)

    if config.dataset_name == "multiomniglot" and not config.data_args["stacked"]:
        waymark_data = tf_spatially_arrange_imgs(waymark_data, config.data_args["n_imgs"], wmark_input=True)

    waymark_data = tf.transpose(waymark_data,[1, 0] + [2 + i for i in range(n_event_dims)])  # (batch_size, n_waymarks, *event_dims)

    return waymark_data


def preprocess_samples_for_dimwise_mixing(positive_samples, negative_samples, n_event_dims,
                                          n_waymarks, event_ones, config):

    if config.dataset_name == "multiomniglot" and not config.data_args["stacked"]:
        # convert spatialomniglot to stackedomniglot - this is just for waymark construction;
        # afterwards we convert all waymark samples to a spatial layout
        positive_samples = tf_unarrange_spatial_imgs(positive_samples, config.data_args["n_imgs"], wmark_input=False)
        negative_samples = tf_unarrange_spatial_imgs(negative_samples, config.data_args["n_imgs"], wmark_input=False)

    positive_samples = tf.tile(tf.expand_dims(positive_samples, axis=0),
                               [n_waymarks, 1] + event_ones)  # (n_waymarks, batch_size, *event_dims)
    negative_samples = tf.tile(tf.expand_dims(negative_samples, axis=0),
                               [n_waymarks, 1] + event_ones)  # (n_waymarks, batch_size, *event_dims)

    if config.n_event_dims_to_mix is not None:
        event_pos = [2 + i for i in range(n_event_dims)]
        transpose_idxs = [0] + event_pos[:-config.n_event_dims_to_mix] + [1] + event_pos[-config.n_event_dims_to_mix:]
        positive_samples = tf.transpose(positive_samples, transpose_idxs)
        negative_samples = tf.transpose(negative_samples, transpose_idxs)

    return positive_samples, negative_samples


def _tf_get_additive_noise(noise_dist, batch_size_per_waymark, config, event_shape, event_ones, n_waymarks, noise_multipliers):

    n_event_dims = len(event_shape)
    b_size = tf.cast(batch_size_per_waymark, tf.int32)
    batch_shape = [b_size]

    if config.noise_dist_name == "flow" and config.create_waymarks_in_zspace:
        noise = tf.reshape(noise_dist.sample_base_dist(batch_shape), batch_shape + event_shape)
    else:
        noise = tf.reshape(noise_dist.sample(batch_shape), batch_shape + event_shape)

    noise = tf.cast(noise, tf.float32)
    noise = tf.tile(tf.expand_dims(noise, axis=-1), [1] + event_ones + [n_waymarks])  # (n, *event_dims, n_waymarks)

    noise *= tf.cast(noise_multipliers, tf.float32)  # (batch_size, *event_dims, n_waymarks)
    noise = tf.transpose(noise, [0, n_event_dims+1] + [i for i in range(1, 1+n_event_dims)])  # (batch_size, n_waymarks, *event_dims)

    return noise


def _tf_get_mults_for_additive_wmarks(waymark_idxs, config, event_ones, waymark_shp):

    noise_multipliers = tf.gather(np.array(config.linear_combo_alphas), waymark_idxs)  # (n_waymarks, )
    noise_multipliers = tf.cast(noise_multipliers, dtype=tf.float32)  # (n_waymarks, )

    data_multipliers = (tf.ones(waymark_shp[1:]) -
                   tf.reshape(noise_multipliers, shape_list(noise_multipliers) + event_ones
                              ) ** 2
                        ) ** 0.5  # (n_waymarks, *event_dims)

    return data_multipliers, noise_multipliers


def _tf_get_shapes_for_additive_wmarks(data, waymark_idxs, config, logger):

    n_data, event_dims = shape_list(data)[0], shape_list(data)[1:]
    n_waymarks = shape_list(waymark_idxs)[0] if shape_list(waymark_idxs) else 1

    if config.shuffle_waymarks:
        logger.info("WARNING: waymarks samples are NOT coupled!")
        n_data -= tf.mod(n_data, n_waymarks)
        batch_size_per_waymark = n_data / n_waymarks
    else:
        logger.info("Waymark samples are coupled! (this is the default setting)")
        batch_size_per_waymark = n_data

    waymark_shp = [batch_size_per_waymark, n_waymarks, *event_dims]
    event_ones = [1 for _ in range(len(event_dims))]  # [1]*num_event_dims

    return batch_size_per_waymark, event_dims, event_ones, n_data, n_waymarks, waymark_shp


def np_build_noise_additive_waymarks_on_the_fly(waymark_idxs, config, data):

    batch_size_per_waymark, event_dims, event_range, event_ones, n_data, n_waymarks, n_wmark_dims, waymark_shp = \
        _np_get_shapes_for_additive_wmarks(waymark_idxs, config, data)

    if config.shuffle_waymarks:
        waymark_data = np.reshape(data[:n_data], waymark_shp)  # (batch_size, n_waymarks, *event_dims)
    else:
        waymark_data = np.tile(np.expand_dims(data, axis=1), [1, n_waymarks] + event_ones)  # (batch_size, n_waymarks, *event_dims)

    multipliers, wmark_stds = _np_get_stds_and_mults_for_additive_wmarks(waymark_idxs, config, event_range, waymark_shp)

    noise = _np_get_additive_noise(batch_size_per_waymark, event_dims, event_range,
                                   event_ones, n_waymarks, n_wmark_dims, wmark_stds)

    waymark_data *= multipliers
    waymark_data += noise

    return waymark_data


def _np_get_additive_noise(batch_size_per_waymark, event_dims, event_range,
                           event_ones, n_waymarks, n_wmark_dims, wmark_stds):

    noise = np.random.normal(size=[batch_size_per_waymark, *event_dims])  # (batch_size, *event_dims)
    noise = np.tile(np.expand_dims(noise, axis=-1), [1] + event_ones + [n_waymarks]) * wmark_stds  # (batch_size, *event_dims, n_waymarks)
    noise = np.transpose(noise, [0, n_wmark_dims-1] + event_range)  # (batch_size, n_waymarks, *event_dims)

    return noise


def _np_get_stds_and_mults_for_additive_wmarks(waymark_idxs, config, event_range, waymark_shp):

    if config.dataset_name == "gaussians":
        multipliers = np.array(config.gauss_interpolation_covariances) / config.gauss_interpolation_covariances[0]
        multipliers = multipliers[waymark_idxs]  # (n_waymarks, )

        wmark_stds = np.sqrt(1 - multipliers)  # (n_waymarks, )
        multipliers = np_expand_multiple_axis(multipliers, axes=event_range)  # (n_waymarks, *event_dims)

    else:
        wmark_stds = np.array(config.init_waymark_coefficients)[waymark_idxs]  # (n_waymarks, )
        data_variances = np.ones(waymark_shp[1:])  # (n_waymarks, *event_dims)
        multipliers = (data_variances - np_expand_multiple_axis(wmark_stds, axes=event_range) ** 2) ** 0.5  # (n_waymarks, *event_dims)

    return multipliers, wmark_stds


def _np_get_shapes_for_additive_wmarks(waymark_idxs, config, data):

    n_data, event_dims = data.shape[0], data.shape[1:]
    n_waymarks = len(waymark_idxs)
    if config.shuffle_waymarks:
        n_data -= np.mod(n_data, n_waymarks)
        batch_size_per_waymark = n_data / n_waymarks
    else:
        batch_size_per_waymark = n_data

    waymark_shp = [batch_size_per_waymark, n_waymarks, *event_dims]
    n_wmark_dims = len(waymark_shp)
    event_range = [i for i in range(1, n_wmark_dims - 1)]
    event_ones = [1 for _ in range(1, n_wmark_dims - 1)]  # [1]*num_event_dims

    return batch_size_per_waymark, event_dims, event_range, event_ones, n_data, n_waymarks, n_wmark_dims, waymark_shp


def build_gauss_sampled_waymarks_on_the_fly(config, logger, n_waymarks, pholders):
    logger.info("--------------------------------------")
    logger.info("sampling gaussian waymarks on-the-fly!")
    logger.info("--------------------------------------")

    correlation_coefs = tf.gather(config.gauss_interpolation_covariances, pholders.waymark_idxs)
    waymark_dists = build_blockwise_correlated_gaussian_waymarks(config.n_dims, n_waymarks, correlation_coefs)
    waymark_data = tf.reshape(waymark_dists.sample(config.n_batch), (config.n_batch, n_waymarks, config.n_dims))

    return AttrDict(locals())


def tf_get_waymark_data(config, pholders, noise_dist=None):
    logger = logging.getLogger("tf")

    if config.waymark_mechanism == "linear_combinations":
        res = tf_build_noise_additive_waymarks_on_the_fly(
            pholders.data, pholders.waymark_idxs, config, logger, noise_dist)

    elif config.waymark_mechanism == "dimwise_mixing":
        res = tf_build_dimwise_mixing_waymarks_on_the_fly(
            pholders.data, pholders.waymark_idxs, pholders.dimwise_mixing_ordering, config, logger, noise_dist)

    else:
        raise ValueError("A method for making waymarks on the fly needs to specified.")

    if config.do_mutual_info_estimation:
        res["waymark0_data"] = (res.waymark_data[0], res.waymark_data[1][:, 0, ...])
    else:
        res["waymark0_data"] = res.waymark_data[:, 0, ...]

    return res


def get_waymark_and_bridge_idxs_for_epoch_i(config, iter_idx):
    logger = logging.getLogger("tf")

    waymark_idxs = config.initial_waymark_indices
    bridge_idxs = waymark_idxs[:-1]

    if config.get("epoch_idx", -1) == 0 and iter_idx == 0:
        logger.info("-------------------------------------")
        logger.info("absolute waymark idxs: {}".format(waymark_idxs))
        logger.info("-------------------------------------")

    config["waymark_idxs"] = waymark_idxs
    config["bridge_idxs"] = bridge_idxs

    return waymark_idxs, bridge_idxs


def get_batch_dimwise_mixing_ordering(batch_shape, config):

    if config.do_mutual_info_estimation:
        batch_shape = [int(batch_shape[0]/2), *batch_shape[1:-1]]

    if config.n_event_dims_to_mix is not None:
        dim_shape = batch_shape[-config.n_event_dims_to_mix:]
    else:
        dim_shape = batch_shape[1:]

    if config.dataset_name == "multiomniglot":
        dim_shape = [config.data_args["n_imgs"]]

    n = batch_shape[0]
    d = np.prod(dim_shape)

    orderings = np.tile(np.arange(1, d + 1), [n, 1])  # (n, d)

    # create per-datapoint dimension orderings
    if config.dimwise_mixing_strategy == "fixed_single_order":
        pass # 'orderings' is already correct

    elif config.dimwise_mixing_strategy == "random_contiguous":
        print("USING CONTIGUOUS MUTATIONS")
        shift = np.random.randint(0, d)
        orderings = np.roll(orderings, shift=shift, axis=1)

    elif config.dimwise_mixing_strategy == "random_non_contiguous":
        print("USING NON-CONTIGUOUS MUTATIONS")
        orderings = seperately_permute_matrix_cols(orderings, transpose=True)  # (n, d)

    return orderings.reshape(n, *dim_shape)
