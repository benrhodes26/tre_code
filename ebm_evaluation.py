from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import tensorflow_probability as tfp

from data_handlers.gaussians import GAUSSIANS
from experiment_ops import build_energies, build_noise_dist, build_data_dist, sample_noise_dist, \
    plot_chains, plot_per_ratio_and_datapoint_diagnostics, CustomMixture, load_flow, load_model, \
    TFCorrelatedGaussians
from mcmc.mcmc_utils import build_mcmc_chain
from mcmc.my_hmc import HamiltonianMonteCarlo
from mcmc.my_langevin import UncalibratedLangevin, EmptyStepSizeAdaptation
from mcmc.my_nuts_v2 import NoUTurnSampler as NoUTurnSampler_v2
from mcmc.my_sample_annealed_importance_chain import sample_annealed_importance_chain
from utils.experiment_utils import *
from utils.misc_utils import *
from utils.plot_utils import *
from utils.tf_utils import *
from waymark_ops import tf_build_noise_additive_waymarks_on_the_fly, tf_build_dimwise_mixing_waymarks_on_the_fly

tfb = tfp.bijectors
tfd = tfp.distributions

# noinspection PyUnresolvedReferences
def build_placeholders():
    if "img_shape" in data_args and data_args["img_shape"] is not None:
        shp = data_args["img_shape"]
        data = tf.placeholder_with_default(np.zeros((1, *shp), dtype=np.float32), (None, *shp), "data")
        waymark_data = tf.placeholder_with_default(np.zeros((1, 1, *shp), dtype=np.float32), (None, None, *shp), "wmark_data")
        initial_states = tf.placeholder_with_default(np.zeros((1, *shp), dtype=np.float32), (None, *shp), "initial_states")
        dimwise_mixing_ordering = tf.placeholder_with_default(np.zeros((1, *shp), dtype=np.int32), (None, *shp), "dimwise_mixing_ordering")
    else:
        data = tf.placeholder_with_default(np.zeros((1, n_dims), dtype=np.float32), (None, n_dims), "data")
        waymark_data = tf.placeholder_with_default(np.zeros((1, 1, n_dims), dtype=np.float32), (None, None, n_dims), "wmark_data")
        initial_states = tf.placeholder_with_default(np.zeros((1, n_dims), dtype=np.float32), (None, n_dims), "initial_states")
        dimwise_mixing_ordering = tf.placeholder_with_default(np.zeros((1, n_dims), dtype=np.int32), (None, n_dims), "dimwise_mixing_ordering")

    single_wmark_idx = tf.placeholder(tf.int32, shape=(None, ), name="single_wmark_idx")
    wmark_sample_size = tf.placeholder(tf.int32, shape=(), name="wmark_sample_size")

    n_steps_per_bridge = tf.placeholder(tf.int32, shape=(None, ), name="n_steps_per_bridge")
    n_leapfrog_steps = tf.placeholder_with_default(ais_n_leapfrog_steps, shape=(), name="n_leapfrog_steps")
    n_leapfrog_steps = tf.cast(n_leapfrog_steps, tf.int32)
    full_model_thinning_factor = tf.placeholder(tf.int32, shape=(), name="full_model_thinning_factor")

    n_noise_samples = tf.placeholder_with_default(ais_n_chains, (), name="n_noise_samples")
    initial_weights = tf.placeholder(tf.float32, shape=(None,), name="initial_raise_weights")
    init_annealed_stepsize = tf.placeholder(tf.float32, shape=(), name="init_annealed_stepsize")
    post_annealed_step_size = tf.placeholder(tf.float32, shape=(), name="post_annealed_step_size")
    post_annealed_n_adapt_steps = tf.placeholder(tf.int32, shape=(), name="post_annealed_n_adapt_steps")
    grad_idx = tf.placeholder(tf.int32, shape=(), name="grad_idx")

    return AttrDict(locals())


# noinspection PyUnresolvedReferences
def build_ais_outer_loop(e_fns,
                         nested_neg_energy_fns,
                         initial_states,
                         sample_method,
                         n_steps_per_bridge,
                         n_leapfrog_steps,
                         forward_mode,
                         initial_step_size,
                         config,
                         initial_weights=None
                         ):
    """Our implementation of AIS has two levels of intermediate distributions:
    an outer level consisting of the bridges in TRE, and an inner level
    using the standard annealed temperature scheme

     Note: reverse AIS (RAISE) can be used by specifiying forward_mode = False
     """
    chains, accept_rates, all_weights = [initial_states], [], []

    cur_states = initial_states
    kernel_results = None
    counter = 0
    step_sizes = []
    nuts_leapfrogs = []

    for i in range(len(e_fns) - 1):
        outer_idx = i+1 if forward_mode else -i-1
        cur_sublist = e_fns[outer_idx]
        cur_nested_sublist = nested_neg_energy_fns[outer_idx]
        prev_energies_fns = [sublist[-1] for sublist in nested_neg_energy_fns[:outer_idx]]

        for j in range(len(cur_sublist)):
            j = j if forward_mode else -j-1

            all_energy_fns = prev_energies_fns + [cur_nested_sublist[j]]
            target_energy_fn = cur_sublist[j]

            initial_weights = initial_weights if counter == 0 else None
            step_idx = -counter - 1 if forward_mode else counter
            n_steps = n_steps_per_bridge[step_idx]

            cur_states, weights, accept_rate, kernel_results = build_ais(initial_state=cur_states,
                                                                         target_energy_fn=target_energy_fn,
                                                                         all_energy_fns=all_energy_fns,
                                                                         sample_method=sample_method,
                                                                         kernel_results=kernel_results,
                                                                         init_step_size=initial_step_size,
                                                                         n_ais_steps=n_steps,
                                                                         n_leapfrog_steps=n_leapfrog_steps,
                                                                         forward_mode=forward_mode,
                                                                         initial_weights=initial_weights,
                                                                         config=config)
            counter += 1

            chains.append(cur_states)
            all_weights.append(weights)
            accept_rates.append(accept_rate)
            step_sizes.append(kernel_results.step_size)
            if sample_method == "nuts":
                nuts_leapfrogs.append(kernel_results.inner_results.inner_results.leapfrogs_taken)

    final_weights = tf.add_n(all_weights)  # (n_chains, )

    # compute AIS log partition / RAISE average log likelihood
    annealing_result = tf.reduce_logsumexp(final_weights) - tf.log(tf.cast(ais_n_chains, tf.float32))

    # calculate variance of the log-weights for all sub-PoEs
    weight_vars = [tf_log_var_exp(tf.add_n(all_weights[:i])) for i in range(1, len(all_weights)+1)]

    res = AttrDict(
        {"annealing_result": annealing_result,
         "chains": tf.stack(chains, axis=1),
         "accept_rates": tf.convert_to_tensor(accept_rates),
         "weight_vars": weight_vars,
         "final_weights": final_weights,
         "step_sizes": tf.convert_to_tensor(step_sizes),
         "nuts_leapfrogs": tf.reduce_mean(tf.convert_to_tensor(nuts_leapfrogs), axis=1)
         }
    )

    return res


# noinspection PyUnresolvedReferences
def build_ais(initial_state,
              target_energy_fn,
              all_energy_fns,
              sample_method,
              kernel_results,
              init_step_size,
              n_ais_steps,
              n_leapfrog_steps,
              config,
              forward_mode=True,
              initial_weights=None
              ):
    """Estimate the log partition of unnormalised model 'target' using annealed importance sampling"""

    n_adapt_steps = config.ais_total_n_steps if config.do_estimate_log_par else config.only_sample_total_n_steps
    if sample_method == "nuts":
        use_mh_step = True
        kernel_fn = lambda tlp_fn, ss: tfp.mcmc.SimpleStepSizeAdaptation(
            # tfp.mcmc.NoUTurnSampler(
            NoUTurnSampler_v2(
                target_log_prob_fn=tlp_fn,
                step_size=ss,
                max_tree_depth=config.ais_nuts_max_tree_depth,
                parallel_iterations=config.parallel_iterations,
                swap_memory=config.swap_memory
            ),
            num_adaptation_steps=n_adapt_steps,
            adaptation_rate=0.05,
            target_accept_prob=0.6,
            step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(step_size=new_step_size),
            step_size_getter_fn=lambda pkr: pkr.step_size,
            log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio
        )

    elif sample_method == "hmc":
        use_mh_step = True
        kernel_fn = lambda tlp_fn, ss: tfp.mcmc.SimpleStepSizeAdaptation(
            # tfp.mcmc.HamiltonianMonteCarlo(
            HamiltonianMonteCarlo(
                target_log_prob_fn=tlp_fn,
                step_size=ss,
                num_leapfrog_steps=n_leapfrog_steps,
                parallel_iterations=config.parallel_iterations,
                swap_memory=config.swap_memory
            ),
            num_adaptation_steps=n_adapt_steps,
            adaptation_rate=0.05,
            target_accept_prob=0.6
        )

    elif sample_method == "metropolis_langevin":
        use_mh_step = True
        kernel_fn = lambda tlp_fn, ss:  tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
                target_log_prob_fn=tlp_fn,
                step_size=ss),
            num_adaptation_steps=n_adapt_steps,
            adaptation_rate=0.05,
            target_accept_prob=0.6
        )

    elif sample_method == "uncalibrated_metropolis_langevin":
        use_mh_step = False
        kernel_fn = lambda tlp_fn, _: EmptyStepSizeAdaptation(
            UncalibratedLangevin(
                target_log_prob_fn=tlp_fn,
                step_size=init_step_size,
                compute_acceptance=False),
        )
    else:
        raise ValueError("must specify a valid mcmc method. `{}' is not a valid choice.".format(sample_method))

    chains, ais_weights, kernel_results = \
        sample_annealed_importance_chain(num_steps=n_ais_steps,
                                         all_energy_fns=all_energy_fns,
                                         target_energy_fn=target_energy_fn,
                                         current_state=initial_state,
                                         make_kernel_fn=kernel_fn,
                                         init_step_size=init_step_size,
                                         kernel_results=kernel_results,
                                         forward=forward_mode,
                                         do_compute_ais_weights=do_estimate_log_par,
                                         initial_weights=initial_weights,
                                         has_accepted_results=False if sample_method == "nuts" else use_mh_step,
                                         parallel_iterations=config.parallel_iterations,
                                         swap_memory=config.swap_memory
                                         )
    if use_mh_step:
        # calculate average acceptance rate across all chains for final step of AIS
        res = kernel_results.inner_results.inner_results
        log_n = tf.log(tf.cast(tf.size(res.log_accept_ratio), res.log_accept_ratio.dtype))
        log_mean_accept_ratio = tf.reduce_logsumexp(tf.minimum(res.log_accept_ratio, 0.)) - log_n
        ais_accept_rate = tf.exp(log_mean_accept_ratio)
    else:
        ais_accept_rate = tf.constant(1.0, tf.float32)

    return chains, ais_weights, ais_accept_rate, kernel_results


def build_model(data, config, invertible_noise=None):

    with tf.compat.v1.variable_scope("tre_model"):

        load_dir = get_metrics_data_dir(config.save_dir, epoch_i=config.eval_epoch_idx)
        loaded_stats = np.load(os.path.join(load_dir, "val.npz"))

        waymark_idxs = loaded_stats["waymark_idxs"]

        max_num_ratios, bridge_idxs = waymark_idxs[-1], waymark_idxs[:-1]
        energy_obj = build_energies(config=config,
                                    bridge_idxs=bridge_idxs,
                                    max_num_ratios=max_num_ratios
                                    )

        neg_energies = energy_obj.neg_energy(data, is_train=False)

        bridge_fns = []
        cum_bridge_fns = []

        total_num_ratios = len(bridge_idxs)
        for i in range(total_num_ratios):

            def single_bridge_fn(x, e=energy_obj, idxs=bridge_idxs, i=i):
                if invertible_noise is not None:
                    x = invertible_noise.forward(x)
                e.bridge_idxs = [idxs[-i-1]]
                return e.neg_energy(x, is_train=False)[:, 0]

            def cumulative_bridge_fn(x, e=energy_obj, idxs=bridge_idxs, i=i):
                if invertible_noise is not None:
                    x = invertible_noise.forward(x)
                e.bridge_idxs = idxs[-i-1:]
                output = e.neg_energy(x, is_train=False)
                return output

            bridge_fns.append(single_bridge_fn)
            cum_bridge_fns.append(cumulative_bridge_fn)

    config.total_num_ratios = total_num_ratios
    config.all_waymark_idxs = waymark_idxs

    return [bridge_fns], [cum_bridge_fns], neg_energies, energy_obj


def build_full_sample(log_prob_fn, initial_states, pholders, config):

    model_samples, model_samples_ar, ss, nuts_leapfrogs_taken \
        = build_mcmc_chain(target_log_prob_fn=log_prob_fn,
                           initial_states=initial_states,
                           n_samples_to_keep=config.post_ais_n_samples_keep,
                           thinning_factor=pholders.full_model_thinning_factor,
                           mcmc_method=config.sample_method,
                           step_size=pholders.post_annealed_step_size,
                           use_adaptive_step_size=True,
                           n_adaptation_steps=pholders.post_annealed_n_adapt_steps,
                           n_leapfrog_steps=pholders.n_leapfrog_steps,
                           nuts_max_tree_depth=config.post_ais_nuts_max_tree_depth,
                           parallel_iterations=config.parallel_iterations,
                           swap_memory=config.swap_memory)

    return model_samples, model_samples_ar, ss, nuts_leapfrogs_taken


def build_data_noise_mixture(noise_dist, data, n_noise_samples, config):

    if config.data_dist_name == "gaussian":
        correlation_coeffient = GAUSSIANS.get_rho_from_mi(config.data_args["true_mutual_info"], config.n_dims)
        data_dist = TFCorrelatedGaussians(config.n_dims, correlation_coeffient)

    elif config.data_dist_name == "flow":
        data_dist = build_data_dist("flow", config, data)

    else:
        raise ValueError("name of target distribution can only be 'gaussian' or 'flow'. "
                         "'{}' is not a valid option.".format(config.data_dist_name))

    noise_data_mixture = CustomMixture(noise_dist, data_dist, 0.5)
    noise_data_mix_samples = noise_data_mixture.sample(n_noise_samples)
    noise_data_mix_log_prob = noise_data_mixture.log_prob(data)

    return data_dist, noise_data_mix_samples, noise_data_mix_log_prob


def sample_from_single_waymark(config, pholders, logger, noise_dist=None):

    if config.waymark_mechanism == "linear_combinations":

        res = tf_build_noise_additive_waymarks_on_the_fly(
            pholders.data, pholders.single_wmark_idx, config, logger, noise_dist)

        waymark_sample = tf.squeeze(res.waymark_data, axis=1)
        waymark_logp = None

    elif config.waymark_mechanism == "dimwise_mixing":
        res = tf_build_dimwise_mixing_waymarks_on_the_fly(
            pholders.data, pholders.single_wmark_idx, pholders.dimwise_mixing_ordering, config, logger, noise_dist)

        waymark_sample = tf.squeeze(res.waymark_data, axis=1)
        waymark_logp = None

    else:
        raise ValueError("A method for making waymarks on the fly needs to specified.")

    return waymark_sample, waymark_logp


# noinspection PyUnresolvedReferences
def build_graph(config):
    """Build graph for computing approximate log partition function via AIS and RAISE

        Returns: dictionary of of all local variables, including the graph ops required for training
    """
    logger = logging.getLogger("tf")
    pholders = build_placeholders()

    # build noise distribution
    d_shape = shape_list(pholders.data)[1:]
    noise_dist = build_noise_dist(noise_dist_name, pholders.data, config, d_shape)
    noise_samples = noise_dist.sample(pholders.n_noise_samples)
    noise_dist_log_prob = noise_dist.log_prob(pholders.data)  # (n_batch, )

    waymark_sample, waymark_log_prob = sample_from_single_waymark(config, pholders, logger, noise_dist)

    if config.data_dist_name:
        data_dist, noise_data_mix_samples, noise_data_mix_log_prob = \
            build_data_noise_mixture(noise_dist, pholders.data, pholders.n_noise_samples, config)

    # build TRE ratio-estimators
    neg_e_fns, nested_neg_e_fns, bridge_neg_energies, energy_obj = build_model(pholders.data, config)

    # insert noise distribution (since it is the first expert in the PoE)
    neg_e_fns.insert(0, [noise_dist.log_prob])
    nested_neg_e_fns.insert(0, [lambda x: tf.expand_dims(noise_dist.log_prob(x), axis=-1)])

    if config.do_sample or config.do_estimate_log_par:
        # build AIS
        ais_results = build_ais_outer_loop(e_fns=neg_e_fns,
                                           nested_neg_energy_fns=nested_neg_e_fns,
                                           initial_states=noise_samples,
                                           sample_method=config.sample_method,
                                           n_steps_per_bridge=pholders.n_steps_per_bridge,
                                           n_leapfrog_steps=pholders.n_leapfrog_steps,
                                           forward_mode=True,
                                           initial_step_size=pholders.init_annealed_stepsize,
                                           config=config)
    if config.do_estimate_log_par:
        # build RAISE
        raise_results = build_ais_outer_loop(e_fns=neg_e_fns,
                                             nested_neg_energy_fns=nested_neg_e_fns,
                                             initial_states=pholders.initial_states,
                                             sample_method=config.sample_method,
                                             n_steps_per_bridge=pholders.n_steps_per_bridge,
                                             n_leapfrog_steps=pholders.n_leapfrog_steps,
                                             forward_mode=False,
                                             initial_weights=pholders.initial_weights,
                                             initial_step_size=pholders.init_annealed_stepsize,
                                             config=config)

    if config.do_post_annealed_sample:
        # sample from the full model, (possibly) starting where annealed sampling finished
        neg_energy_fns = [sublist[-1] for sublist in nested_neg_e_fns][::-1]  # order from data --> noise dist
        def model_log_prob_fn(x):
            return tf.add_n([tf.reduce_sum(f(x), axis=-1) for f in neg_energy_fns])

        model_samples, model_samples_ar, model_stepsize, nuts_leapfrogs_taken = \
            build_full_sample(model_log_prob_fn, pholders.initial_states, pholders, config)

    # calculate neg_energy contribution from each bridge and the noise distribution
    bridges_plus_noise_logp = tf.concat([bridge_neg_energies,
                                        tf.expand_dims(noise_dist_log_prob, axis=1)], axis=1)  # (n, n_ratios+1)

    # calculate (unnormalised) log likelihood
    prenorm_logliks = tf.reduce_sum(bridges_plus_noise_logp, axis=-1)  # (n, )

    av_submodel_grads = maybe_build_gradient_wrt_input(config, neg_e_fns, pholders)

    graph = AttrDict(locals())
    graph.update(pholders)
    return graph   # dict whose values can be accessed as attributes i.e. val = dict.key


# noinspection PyUnresolvedReferences
def build_flow_based_graph(config):
    """Build graph for computing approximate log partition function via AIS and RAISE

    All sampling computations are done in the z-space of a flow, and then mapped back to x-space

        Returns: dictionary of of all local variables, including the graph ops required for training
    """
    logger = logging.getLogger("tf")
    pholders = build_placeholders()

    # build noise distribution
    d_shape = shape_list(pholders.data)[1:]
    noise_dist = build_noise_dist(noise_dist_name, pholders.data, config, d_shape)
    noise_samples = noise_dist.sample(pholders.n_noise_samples)
    noise_dist_log_prob = noise_dist.log_prob(pholders.data)  # (n_batch, )

    waymark_sample, waymark_log_prob = sample_from_single_waymark(config, pholders, logger, noise_dist)

    if config.data_dist_name:
        data_dist, noise_data_mix_samples, noise_data_mix_log_prob = \
            build_data_noise_mixture(noise_dist, pholders.data, pholders.n_noise_samples, config)

    # build base dist of flow
    base_dist = noise_dist.base_dist
    base_samples = base_dist.sample(pholders.n_noise_samples)

    # build TRE ratio-estimators
    neg_e_fns, nested_neg_e_fns, bridge_neg_energies, energy_obj = \
        build_model(pholders.data, config, invertible_noise=noise_dist)

    # insert flow base dist (since it is the first expert in the z-space PoE)
    neg_e_fns.insert(0, [base_dist.log_prob])
    nested_neg_e_fns.insert(0, [lambda x: tf.expand_dims(base_dist.log_prob(x), axis=-1)])

    if config.do_sample or config.do_estimate_log_par:
        # build AIS
        ais_results = build_ais_outer_loop(e_fns=neg_e_fns,
                                           nested_neg_energy_fns=nested_neg_e_fns,
                                           initial_states=base_samples,
                                           sample_method=config.sample_method,
                                           n_steps_per_bridge=pholders.n_steps_per_bridge,
                                           n_leapfrog_steps=pholders.n_leapfrog_steps,
                                           forward_mode=True,
                                           initial_step_size=pholders.init_annealed_stepsize,
                                           config=config)
        ais_results.chains = noise_dist.forward(ais_results.chains, collapse_wmark_dims=True)

    z_space_init_states = noise_dist.inverse(pholders.initial_states)

    if config.do_estimate_log_par:
        # build RAISE
        raise_results = build_ais_outer_loop(e_fns=neg_e_fns,
                                             nested_neg_energy_fns=nested_neg_e_fns,
                                             initial_states=z_space_init_states,
                                             sample_method=config.sample_method,
                                             n_steps_per_bridge=pholders.n_steps_per_bridge,
                                             n_leapfrog_steps=pholders.n_leapfrog_steps,
                                             forward_mode=False,
                                             initial_weights=pholders.initial_weights,
                                             initial_step_size=pholders.init_annealed_stepsize,
                                             config=config)
        raise_results.chains = noise_dist.forward(raise_results.chains, collapse_wmark_dims=True)

    if config.do_post_annealed_sample:
        # sample from the full model, (possibly) starting where annealed sampling finished
        neg_energy_fns = [sublist[-1] for sublist in nested_neg_e_fns][::-1]  # order from data --> noise dist
        def model_log_prob_fn(x):
            return tf.add_n([tf.reduce_sum(f(x), axis=-1) for f in neg_energy_fns])

        model_samples, model_samples_ar, model_stepsize, nuts_leapfrogs_taken = \
            build_full_sample(model_log_prob_fn, z_space_init_states, pholders, config)

        model_samples = noise_dist.forward(model_samples, collapse_wmark_dims=True)

    # eval neg_energy contribution of each ratio in x-space
    bridges_plus_noise_logp = tf.concat([bridge_neg_energies,
                                         tf.expand_dims(noise_dist_log_prob, axis=1)], axis=1)  # (n, n_ratios+1)

    # calculate (unnormalised) log likelihood in x-space
    prenorm_logliks = tf.reduce_sum(bridges_plus_noise_logp, axis=-1)  # (n, )

    av_submodel_grads = maybe_build_gradient_wrt_input(config, neg_e_fns, pholders, flow_inv_fn=noise_dist.inverse)

    graph = AttrDict(locals())
    graph.update(pholders)
    return graph   # dict whose values can be accessed as attributes i.e. val = dict.key


def maybe_build_gradient_wrt_input(config, neg_e_fns, pholders, flow_inv_fn=None):
    if config.do_assess_subbridges:
        if (config.data_args is not None) and ("img_shape" in config.data_args) and (config.data_args["img_shape"] is not None):
            event_shp = config.data_args["img_shape"]
        else:
            event_shp = [config.n_dims]
        b_size = config.n_batch // config.total_num_ratios

        assign_val = pholders.data[:b_size]
        if flow_inv_fn is not None:
            assign_val = flow_inv_fn(assign_val)
            event_shp = [np.prod(np.array(event_shp))]

        grad_input_var = tf.get_variable('grad_input_var', shape=[b_size, *event_shp], dtype=tf.float32)
        assign_input = tf.assign(grad_input_var, assign_val)

        with tf.control_dependencies([assign_input]):
            bridge_terms = [e(grad_input_var) for sublist in neg_e_fns for e in sublist]  # flatten nested list
            per_bridge_grads = [tf.gradients(y, grad_input_var)[0] for y in bridge_terms]
            av_bridge_grads = [tf.reduce_mean(g, axis=0) for g in per_bridge_grads]

        gather_idxs = tf.range(0, len(av_bridge_grads) - pholders.grad_idx)
        av_bridge_grads = tf.gather(av_bridge_grads, gather_idxs)  # list of data.shape tensors
        av_submodel_grads = tf.reduce_sum(av_bridge_grads, axis=0)  # data.shape
    else:
        av_submodel_grads = tf.no_op()

    return av_submodel_grads


def estimate_gauss_covar(samples, true_cov_matrix, conf, name="direct"):
    """Estimate covariance from samples and compare to ground truth"""
    if len(samples.shape) == 3:
        n, k, d = samples.shape
        samples = samples.reshape(-1, d)  # combine all samples from mcmc chains

    cov_matrix = np.cov(samples, rowvar=False)  # (d, d)

    deltas = np.abs(true_cov_matrix - cov_matrix)
    mse = np.mean(deltas)

    # mse of non-zero entries
    non_zero_idxs = np.abs(true_cov_matrix) > 1e-4
    nonzero_deltas = np.abs(true_cov_matrix[non_zero_idxs] - cov_matrix[non_zero_idxs])
    nonzero_mse = np.mean(nonzero_deltas)

    # analytic cross-entropy between true gauss & model
    # note that means of both gaussians are zero, simplifying the computation
    cross_entropy = cross_entropy_two_gaussians(true_cov_matrix, cov_matrix)
    estimated_mi = -cross_entropy - conf["noise_dist_loglik"]

    logger = logging.getLogger("tf")
    logger.info("{} Gaussian results...".format(name))
    logger.info("mse of entire estimated covariance matrix is {}".format(mse))
    logger.info("mse of non-zero entries is {}".format(nonzero_mse))
    logger.info("estimated mutual info {}".format(estimated_mi))
    conf["{}_gauss_mse".format(name)] = mse
    conf["{}_gauss_nonzero_mse".format(name)] = nonzero_mse
    conf["{}_gauss_kl".format(name)] = estimated_mi


# noinspection PyUnresolvedReferences
def evaluate_energies_and_losses(graph, sess, val_dp, ais_save_dir, config, logger):

    # skip this method if we're not interested in estimating log partition fn
    if not config.do_estimate_log_par:
        config["prenormalised_kl"] = 0.0
        config["prenormalised_js"] = 0.0
        config["noise_dist_loglik"] = 0.0
        config["prenormalised_loglik"] = 0.0
        config["dv_bound"] = 0.0
        config["nwj_bound"] = 0.0
        return np.zeros(len(val_dp.data))

    full_model_logp_wrt_data, full_model_logp_wrt_noise, log_IS_weight = \
        evaluate_energies(ais_save_dir, config, val_dp, graph, sess)

    # extract the neg energies of the bridges and the av. loglik of noise distribution w.r.t data
    bridge_logps_wrt_data = full_model_logp_wrt_data[:, :-1]  # (n, n_ratios)
    bridge_logps_wrt_noise = full_model_logp_wrt_noise[:, :-1]  # (n, n_ratios)
    noise_dist_loglik = np.mean(full_model_logp_wrt_data[:, -1])

    evaluate_losses(bridge_logps_wrt_data, bridge_logps_wrt_noise, noise_dist_loglik,
                    log_IS_weight, ais_save_dir, config, logger)

    prenormalised_logp = np.sum(full_model_logp_wrt_data, axis=1)  # (n, )

    return prenormalised_logp


def evaluate_losses(bridge_logps_wrt_data, bridge_logps_wrt_noise, noise_dist_loglik, e2, ais_save_dir, config, logger):

    # evaluate model under the DV & NWJ losses (defined in http://proceedings.mlr.press/v97/poole19a/poole19a.pdf)
    e1 = np.sum(bridge_logps_wrt_data, axis=1)
    e2 += np.sum(bridge_logps_wrt_noise, axis=1)
    is_var = log_var_exp(e2)  # log(variance of importance sampling weights)

    dv, dv_term1, dv_term2 = dv_bound_fn(e1, e2)
    nwj, nwj_term1, nwj_term2 = nwj_bound_fn(e1, e2)

    prenorm_kl = np.mean(e1)
    prenorm_js = jensen_shannon_fn(e1, e2, 0.0)
    prenorm_loglik = prenorm_kl + noise_dist_loglik

    logger.info("prenorm_kl: {:.2f} | prenorm_loglik: {:.2f}".format(prenorm_kl, prenorm_loglik))
    logger.info("dv bound: {:.2f} | nwj_bound: {:.2f}".format(dv, nwj))
    logger.info("log of variance of IS weights: {}".format(is_var))
    np.savetxt(os.path.join(ais_save_dir, "dv_nwj_lower_bounds"),
               np.array([dv, nwj, is_var]),
               header="dv_bound/nwj_bound/log_IS_weights_var")

    config["prenormalised_kl"] = prenorm_kl
    config["prenormalised_js"] = prenorm_js
    config["noise_dist_loglik"] = noise_dist_loglik
    config["prenormalised_loglik"] = prenorm_loglik
    config["dv_bound"] = dv
    config["nwj_bound"] = nwj


def evaluate_energies(ais_save_dir, config, val_dp, graph, sess):

    # evaluate neg energies of data samples (and plot diagnostics)
    bridges_plus_noise_logp1 = plot_per_ratio_and_datapoint_diagnostics(sess=sess,
                                                                        metric_op=graph.bridges_plus_noise_logp,
                                                                        num_ratios=config.total_num_ratios,
                                                                        datasets=[val_dp.data],
                                                                        data_splits=["val"],
                                                                        save_dir=ais_save_dir,
                                                                        dp=val_dp,
                                                                        config=config,
                                                                        data_pholder=graph.data, name="neg_e_data")

    # evaluate neg energies of noise samples (and plot diagnostics)
    n_samples = config.n_noise_samples_for_variational_losses
    if config.data_dist_name:
        samples = tf_batched_operation(sess=sess,
                                       ops=graph.noise_data_mix_samples,
                                       n_samples=n_samples,
                                       batch_size=min(1000, n_samples),
                                       const_feed_dict={graph.n_noise_samples: min(1000, n_samples)})
        logp_1, logp_2 = tf_batched_operation(sess=sess,
                                              ops=[graph.noise_dist_log_prob, graph.noise_data_mix_log_prob],
                                              n_samples=samples.shape[0],
                                              batch_size=min(1000, samples.shape[0]),
                                              data_pholder=graph.data,
                                              data=samples)
        log_IS_weight = logp_1 - logp_2

    else:
        samples = sample_noise_dist(sess, graph, config.noise_dist_name, val_dp, n_samples)
        log_IS_weight = 0

    bridges_plus_noise_logp2 = plot_per_ratio_and_datapoint_diagnostics(sess=sess,
                                                                        metric_op=graph.bridges_plus_noise_logp,
                                                                        num_ratios=config.total_num_ratios,
                                                                        datasets=[samples],
                                                                        data_splits=["noise_dist_samples"],
                                                                        save_dir=ais_save_dir, dp=val_dp, config=config,
                                                                        data_pholder=graph.data,
                                                                        name="neg_e_noise_samples")

    return bridges_plus_noise_logp1, bridges_plus_noise_logp2, log_IS_weight


# noinspection PyUnresolvedReferences
def run_annealing_methods(g,
                          sess,
                          val_dp,
                          prenormalised_logp,
                          ais_save_dir,
                          config):

    logger = logging.getLogger("tf")

    total_n_steps = config.ais_total_n_steps if config.do_estimate_log_par else config.only_sample_total_n_steps
    n_steps_per_bridge = np.ones(config.total_num_ratios) * (1 / config.total_num_ratios) * total_n_steps
    logger.info("number of mcmc steps per bridge: {}".format(n_steps_per_bridge))

    # note: if do_estimate_log_par == False, but do_sample==True, then we still make a call to 'run_ais'.
    # The log partition will not actually be estimated, but we use exactly the same annealing procedure as AIS to
    # obtain samples from the model (e.g via sampling along a path that interpolates between the noise distribution and the model)
    ais_final_states, ais_final_step_size = run_ais(g=g,
                                                    sess=sess,
                                                    prenormalised_logp=prenormalised_logp,
                                                    n_steps_per_bridge=n_steps_per_bridge,
                                                    ais_save_dir=ais_save_dir,
                                                    val_dp=val_dp,
                                                    config=config)
    if config.do_estimate_log_par:
        run_raise(g=g,
                  sess=sess,
                  prenormalised_logp=prenormalised_logp,
                  init_step_size=ais_final_step_size,
                  n_steps_per_bridge=n_steps_per_bridge,
                  val_dp=val_dp,
                  ais_save_dir=ais_save_dir,
                  config=config)

    return ais_final_states, ais_final_step_size


# noinspection PyUnresolvedReferences
def run_ais(g,
            sess,
            prenormalised_logp,
            n_steps_per_bridge,
            ais_save_dir,
            val_dp,
            config):

    logger = logging.getLogger("tf")
    if config.do_estimate_log_par:
        logger.info("Running AIS...")
        n_chains = config.ais_n_chains
    else:
        logger.info("Sampling from the model via annealing")
        n_chains = config.only_sample_n_chains

    pre_ais_time = time()

    fd = {g.n_steps_per_bridge: n_steps_per_bridge,
          g.n_noise_samples: n_chains,
          g.init_annealed_stepsize: config.ais_step_size_init}

    ais_results = AttrDict(sess.run(g.ais_results, feed_dict=fd))

    ais_time = time() - pre_ais_time
    logger.info("AIS finished. Took {} seconds".format(ais_time))

    ais_log_partition = ais_results.annealing_result
    prenormalised_av_ll = np.mean(prenormalised_logp)
    ais_av_ll = prenormalised_av_ll - ais_log_partition

    summarise_annealing_results(g, sess, ais_av_ll, ais_results, "ais", ais_save_dir, val_dp, config)

    ais_final_states = ais_results.chains[:, -1, ...]
    try:
        ais_final_step_size = ais_results.step_sizes[-1][-1]
    except:
        ais_final_step_size = ais_results.step_sizes[-1]

    return ais_final_states, ais_final_step_size


def run_raise(g,
              sess,
              prenormalised_logp,
              init_step_size,
              n_steps_per_bridge,
              val_dp,
              ais_save_dir,
              config):

    logger = logging.getLogger("tf")
    logger.info("Running RAISE...")

    fd = {g.initial_states: val_dp.data[:config.ais_n_chains],
          g.n_steps_per_bridge: n_steps_per_bridge,
          g.initial_weights: prenormalised_logp[:config.ais_n_chains],
          g.init_annealed_stepsize: init_step_size}


    raise_results = AttrDict(sess.run(g.raise_results, feed_dict=fd))

    raise_log_probs = raise_results.final_weights  # final (log) weights
    cv_raise_log_probs = raise_log_probs - prenormalised_logp[:config.ais_n_chains]  # control variate
    logger.info("raise log prob std: {} \n "
                "after control variate: {}".format(np.std(raise_log_probs), np.std(cv_raise_log_probs)))

    if np.std(cv_raise_log_probs) <= np.std(raise_log_probs):
        raise_av_ll = np.mean(cv_raise_log_probs) + np.mean(prenormalised_logp)
    else:
        raise_av_ll = np.mean(raise_log_probs)

    summarise_annealing_results(g, sess, raise_av_ll, raise_results, "raise", ais_save_dir, val_dp, config)


def summarise_annealing_results(graph, sess, normalised_av_ll, results, name, save_dir, dp, config):
    logger = logging.getLogger("tf")

    bits_per_dim = convert_to_bits_per_dim(normalised_av_ll + np.mean(dp.source.ldj), config.n_dims, dp.source.original_scale)

    logger.info("{} av loglik : {:.2f}".format(name, normalised_av_ll))
    logger.info("{} bits per dim : {:.2f}".format(name, bits_per_dim))
    logger.info("{} final_step_sizes are {}".format(name, results.step_sizes))
    if config.sample_method == "nuts":
        logger.info("{} total nuts leapfrogs steps: {}".format(name, results.nuts_leapfrogs))

    config["{}_loglik".format(name)] = normalised_av_ll
    config["{}_bits_per_dim".format(name)] = bits_per_dim
    config["{}_kl".format(name)] = normalised_av_ll - config["noise_dist_loglik"]
    config["{}_weight_vars".format(name)] = results.weight_vars[-1]
    config["{}_final_step_sizes".format(name)] = [i for i in results.step_sizes.flatten()]
    if config.sample_method == "nuts":
        config["{}_num_nuts_leapfrogs".format(name)] = [i for i in results.nuts_leapfrogs]

    np.savez_compressed(save_dir + "{}_chains".format(name), samples=results.chains)
    np.savez_compressed(save_dir + "{}_weights".format(name), weights=results.final_weights)

    plotscatter_one_per_axis(
        x=[results.accept_rates, results.weight_vars],
        xlabels=["state_idx"]*2,
        ylabels=["{}_acceptance_rates".format(name), "{}_log_variance_of_weights".format(name)],
        dir_name=save_dir,
        name="{}_metrics".format(name)
    )

    plot_chains(chains=results.chains,
                name=name,
                save_dir=save_dir,
                dp=dp,
                config=config,
                graph=graph,
                sess=sess,
                rank_op=graph.prenorm_logliks,
                plot_hists=config.data_dist_name == "gaussian",
                is_annealed_samples=True)

    plot_sample_diagnostics(sess, graph, results.chains, name, save_dir, dp, config)

    # evaluate neg energies of final states of chain
    if name == "ais":
        try:
            plot_per_ratio_and_datapoint_diagnostics(sess=sess,
                                                     metric_op=graph.bridges_plus_noise_logp,
                                                     num_ratios=config.total_num_ratios,
                                                     datasets=[results.chains[:, -1, ...]],
                                                     data_splits=["ais_samples"],
                                                     save_dir=save_dir,
                                                     dp=dp, config=config,
                                                     data_pholder=graph.data,
                                                     name="ais_samples")
        except ValueError as e:
            logger.info("plotting neg energies hists of samples failed. Error: {}".format(e))

    logger.info("Finished {}".format(name))



def run_full_model_samplers(sess, graph, post_annealed_initial_states, ais_final_step_size, val_dp, ais_save_dir, config, logger):

    logger.info("Running full-model MCMC sampler...")

    run_full_model_sampling(g=graph,
                            sess=sess,
                            sample_op=graph.model_samples,
                            accept_rate_op=graph.model_samples_ar,
                            step_size_op=graph.model_stepsize,
                            init_states=post_annealed_initial_states,
                            step_size=ais_final_step_size,
                            thinning_factor=config.post_ais_thinning_factor,
                            val_dp=val_dp,
                            ais_save_dir=ais_save_dir,
                            config=config,
                            name="{}_post_ais".format(config.sample_method))


# noinspection PyUnresolvedReferences
def run_full_model_sampling(g,
                            sess,
                            sample_op,
                            accept_rate_op,
                            step_size_op,
                            init_states,
                            step_size,
                            thinning_factor,
                            val_dp,
                            ais_save_dir,
                            config,
                            name):
    logger = logging.getLogger("tf")
    n_adapt_steps = int(config.post_ais_n_samples_keep * max(thinning_factor, 1) / 2)
    n_samples_to_discard = int(n_adapt_steps / max(thinning_factor, 1))

    fd = {
        g.initial_states: init_states,
        g.full_model_thinning_factor: thinning_factor,
        g.post_annealed_step_size: step_size,
        g.post_annealed_n_adapt_steps: n_adapt_steps
    }
    ops = [sample_op, accept_rate_op, step_size_op, g.nuts_leapfrogs_taken]

    # Run MCMC on the full TRE model
    all_chains, accept_rate, final_ss, nuts_leapfrogs = sess.run(ops, feed_dict=fd)

    final_samples = all_chains[:, n_samples_to_discard:, ...].reshape(-1, *all_chains.shape[2:])

    logger.info("{} MCMC sampling:".format(name))
    logger.info("Final acceptance rate: {}".format(accept_rate))
    logger.info("Final step size: {}".format(final_ss[0]))
    if config.sample_method == "nuts": logger.info("Num nuts leapfrogs: {}".format(nuts_leapfrogs))

    config["{}_accept_rates".format(name)] = accept_rate
    config["{}_final_stepsizes".format(name)] = [i for i in final_ss[0]]
    if config.sample_method == "nuts":
        config["{}_num_nuts_leapfrogs".format(name)] = [i for i in nuts_leapfrogs]

    logger.info("saving chains to disk...")
    np.savez_compressed(ais_save_dir + "{}_chains".format(name), samples=all_chains)

    # create various plots to analyse the chains
    logger.info("plotting chains...")
    plot_chains(all_chains,
                "{}_samples".format(name),
                ais_save_dir,
                dp=val_dp,
                config=config,
                graph=g,
                sess=sess,
                rank_op=g.prenorm_logliks,
                plot_hists=config.data_dist_name == "gaussian")

    logger.info("plotting sample diagnostics...")
    plot_sample_diagnostics(sess, g, all_chains, name, ais_save_dir, val_dp, config)

    if config.dataset_name == "gaussians":
        logger.info("Estimating gausian covariance matrix from data & samples...")
        true_cov = val_dp.source.cov_matrix
        estimate_gauss_covar(val_dp.data[:len(final_samples)], true_cov, config, "direct")
        estimate_gauss_covar(final_samples, true_cov, config, "indirect")


def assess_bridges(sess, g, dp, config, which_set="train"):
    """Evaluate bridges under different waymark distributions, and visualise the results in various ways"""

    fig_dir = os.path.join(config.save_dir, "figs/subbridges/")
    os.makedirs(fig_dir, exist_ok=True)

    b_size = config.n_batch // config.total_num_ratios
    sample_size = min(len(dp.data), 1000)
    sample_size = sample_size - sample_size % b_size
    data = dp.data[:sample_size]
    num_waymarks = len(config.all_waymark_idxs)

    bridge_waymark_grid = np.zeros((num_waymarks, sample_size, num_waymarks))  # (n_waymarks, sample_size, n_bridges+1)
    submodel_norm_of_grads = np.zeros(num_waymarks)  # (n_waymarks, )
    for i, wmark_idx in enumerate(config.all_waymark_idxs):

        waymark = tf_batched_operation(
            sess,
            ops=[g.waymark_sample],
            n_samples=sample_size,
            batch_size=min(len(dp.data), 100),
            data_pholder=g.data,
            data=data,
            const_feed_dict={g.single_wmark_idx: [wmark_idx],
                             g.wmark_sample_size: min(len(dp.data), 100)}
        )

        bridge_waymark_grid[i], sub_grads = \
            tf_batched_operation(sess,
                                 ops=[g.bridges_plus_noise_logp, g.av_submodel_grads],
                                 n_samples=sample_size,
                                 batch_size=b_size,
                                 data_pholder=g.data,
                                 data=waymark,
                                 const_feed_dict={g.grad_idx: i})
        submodel_norm_of_grads[i] = np.linalg.norm(np.mean(sub_grads, axis=0)) / config.n_dims

    make_bridge_plots(bridge_waymark_grid, fig_dir, which_set)

    fig, ax = plt.subplots(1, 1)
    plotscatter_single_axis(ax, submodel_norm_of_grads, "waymark_model_idx", "norm_of_expected_grad")
    save_fig(fig_dir, "{}_norm_of_grad_of_wmark_logp".format(which_set))
    plot_impsamp_normconsts(bridge_waymark_grid, fig_dir, which_set)


def assess_parameters(config):
    """Compute various statistics of the parameters of the network"""

    ais_save_dir = get_ais_dir()
    param_dir = path_join(ais_save_dir, "parameter_stats/")
    all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    # compute means & stds of per-ratio scales + biases
    if config.network_type != "quadratic":
        if config.network_type == "mlp":
            per_bridge_scales = [v.eval() for v in all_vars if "scale" in v.name and "cond_scale_shift" in v.name]
            per_bridge_biases = [v.eval() for v in all_vars if "bias" in v.name and "cond_scale_shift" in v.name]
        else:
            per_bridge_scales = [v.eval() for v in all_vars if "gamma" in v.name and "cond_conv" in v.name]
            per_bridge_biases = [v.eval() for v in all_vars if "beta" in v.name and "cond_conv" in v.name]

        plot_mean_std_param_stats(per_bridge_scales, "per_bridge_scales", param_dir)
        plot_mean_std_param_stats(per_bridge_biases, "per_bridge_biases", param_dir)

    # compute singular values of final quadratic head params
    L = [v for v in all_vars if "Q_all" in v.name][0]
    L = tf_enforce_lower_diag_and_nonneg_diag(L, shift=5.0)
    Q = tf.matmul(L, tf.transpose(L, [0, 2, 1]))  # (num_ratios, input_dim, input_dim)
    plot_head_singular_values(Q.eval(), "quadratic_heads_matrices", param_dir)

    # compute singular values of final linear head params
    W = [v.eval() for v in all_vars if "W_all" in v.name][0]
    plot_head_singular_values(W, "linear_heads_matrices", param_dir)

    norm_consts = [v.eval() for v in all_vars if "b_all" in v.name][0]
    fig, ax = plt.subplots(1, 1)
    plotscatter_single_axis(ax, norm_consts, "bridge_idxs", "scale_param")
    save_fig(param_dir, "scale_params")

    # assess spectral norms of each weight matrix
    # kernels = [v.eval() for v in all_vars if "kernel" in v.name]


def plot_head_singular_values(mats, name, ais_dir):

    singular_vals = np.linalg.svd(mats, compute_uv=False)

    fig, axs = plt.subplots(len(singular_vals.shape), 1)
    axs = axs.ravel() if isinstance(axs, np.ndarray) else [axs]
    mult_plotscatter_single_axis(axs[0], singular_vals, "singular_value_idx", "singular_value_magnitude",
                                 labels=[str(i) for i in range(len(singular_vals))])
    if len(singular_vals.shape) == 2:
        mult_plotscatter_single_axis(axs[1], singular_vals[:, :10], "singular_value_idx", "singular_value_magnitude",
                                     labels=[str(i) for i in range(len(singular_vals))])
    fig.legend()
    save_fig(ais_dir, name)


def plot_mean_std_param_stats(X, name, ais_save_dir):

    reduce_ax = np.arange(1, len(X[0].shape), dtype=np.int32)
    x_mean = np.array([np.mean(x, axis=tuple(reduce_ax)) for x in X])
    x_std = np.array([np.std(x, axis=tuple(reduce_ax)) for x in X])

    fig, axs = plt.subplots(2, 1)
    axs = axs.ravel()

    mult_plotscatter_single_axis(axs[0], x_mean, "bridge_idx", "{}_mean".format(name),
                                 labels=[str(i) for i in range(len(x_mean))])
    mult_plotscatter_single_axis(axs[1], x_std, "bridge_idx", "{}_std".format(name),
                                 labels=[str(i) for i in range(len(x_mean))])

    fig.legend()
    save_fig(ais_save_dir, name)


def plot_impsamp_normconsts(bridge_waymark_grid, fig_dir, which_set):
    num_waymarks = bridge_waymark_grid.shape[-1]
    IS_normcon_of_waymarks_with_data_proposal = log_mean_exp(np.cumsum(-bridge_waymark_grid[0, :, :-1], axis=-1), axis=0)
    IS_normcon_of_waymarks_with_noise_proposal = log_mean_exp(np.cumsum(bridge_waymark_grid[-1, :, -2::-1], axis=-1)[::-1], axis=0)

    fig, ax = plt.subplots(2, 1)
    ax = ax.ravel()
    plotscatter_single_axis(ax[0], IS_normcon_of_waymarks_with_data_proposal,
                            "waymark_idx", "estimated norm const",
                            "imp sample estimated norm consts with data proposal",
                            np.arange(1, num_waymarks))
    plotscatter_single_axis(ax[1], IS_normcon_of_waymarks_with_noise_proposal,
                            "waymark_idx", "estimated norm const",
                            "imp sample estimated norm consts with noise proposal",
                            np.arange(num_waymarks-1))
    fig.tight_layout()
    save_fig(fig_dir, "{}_IS_estimates_of_each_waymark_normalising_const".format(which_set))


def plot_sample_diagnostics(sess, graph, all_chains, name, ais_save_dir, dp, config):
    n_chains, n_states = all_chains.shape[:2]
    final_samples = all_chains[:, -1, ...].reshape(n_chains, -1)

    plotscatter_one_per_axis(
        x=five_stat_sum(final_samples, axis=0),
        xlabels=["state_idx"] * 5,
        ylabels=["mean", "median", "std", "min", "max"],
        dir_name=ais_save_dir,
        name="samples_five_stat_sum",
        title="Final annealed samples summary stats"
    )

    plot_hist(np.linalg.norm(final_samples, axis=1), dir_name=ais_save_dir, name="norm_of_final_samples_hist")
    plot_hist(np.mean(final_samples, axis=1), dir_name=ais_save_dir, name="mean_of_final_samples_hist")

    neg_energies_grid = np.zeros((n_states, n_chains, config.total_num_ratios + 1))
    for i in range(n_states):
        neg_energies_grid[i] = tf_batched_operation(sess,
                                                    ops=[graph.bridges_plus_noise_logp],
                                                    n_samples=n_chains,
                                                    batch_size=min(config.n_batch // config.total_num_ratios, n_chains),
                                                    data_pholder=graph.data,
                                                    data=all_chains[:, i, ...])

    plot_per_bridge_energy_graphs(neg_energies_grid,
                                  ais_save_dir + "subbridges_{}/".format(name),
                                  name,
                                  which_set="mcmc",
                                  index_name="state")

    plot_per_ratio_and_datapoint_diagnostics(sess=sess,
                                             metric_op=graph.bridges_plus_noise_logp,
                                             num_ratios=config.total_num_ratios,
                                             datasets=[all_chains[:, -1, ...]],
                                             data_splits=["{}_samples".format(name)],
                                             save_dir=ais_save_dir, dp=dp,
                                             config=config,
                                             data_pholder=graph.data,
                                             name="{}_samples".format(name))


def make_bridge_plots(bridge_waymark_grid, fig_dir, which_set):

    # plot histograms of each bridge evaluated at it's 2 respective waymarks
    plot_neg_energy_hists_per_bridge_evaled_at_own_waymarks(bridge_waymark_grid, fig_dir, which_set)

    # for each bridge, plot its average value when evaulated at each waymark distribution
    num_waymarks = plot_per_bridge_energy_graphs(bridge_waymark_grid, fig_dir, "Each bridge at each waymark", which_set)

    # compute all n(n-1)/2 classification losses, and plot these in a grid
    plot_all_nsquared_losses(bridge_waymark_grid, fig_dir, num_waymarks, which_set)

    # plot scatter of logp_noise(x) vs r(x), where x~p_data
    plot_noise_vs_combined_ratio(bridge_waymark_grid, fig_dir, which_set)


def plot_noise_vs_combined_ratio(bridge_waymark_grid, fig_dir, which_set):

    noise_vals = bridge_waymark_grid[0, :, -1]  # (n_samples, )
    ratio_vals = np.sum(bridge_waymark_grid[0, :, :-1], axis=-1)  # (n_samples, )

    fig, ax = plt.subplots(1, 1)
    ax.scatter(noise_vals, ratio_vals)
    ax.set_xlabel(r"$\log p_{noise} (\mathbf{x})$," + "   " + r"$\mathbf{x} \sim p_{data}$")
    ax.set_ylabel(r"$\log r (\mathbf{x})$" + "   " + r"$\mathbf{x} \sim p_{data}$")
    fig.legend()
    save_fig(fig_dir, "scatter_noise_vs_ratio_at_data_{}".format(which_set))


def plot_all_nsquared_losses(bridge_waymark_grid, fig_dir, num_waymarks, which_set):
    # Compute all n(n-1)/2 classification losses
    loss_grid, separate_term_loss_grid = np.zeros((num_waymarks, num_waymarks)), np.zeros((num_waymarks, num_waymarks))
    for i in range(num_waymarks - 1):
        compute_loss_wmark_i_vs_j_greater_than_i(loss_grid, separate_term_loss_grid, bridge_waymark_grid[:, :, :-1], i)

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(loss_grid)
    ax.set_xlabel("waymark j")
    ax.set_ylabel("waymark i")
    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    save_fig(fig_dir, "{}_all_n_squared_losses".format(which_set))


def plot_neg_energy_hists_per_bridge_evaled_at_own_waymarks(bridge_waymark_grid, fig_dir, which_set):

    n_bridges = len(bridge_waymark_grid) - 1
    axs, fig = create_subplot_with_max_num_cols(n_bridges, max_n_cols=4)
    for i in range(n_bridges):
        ax = axs[i]

        bridge_i_numer = bridge_waymark_grid[i, :, i]
        bridge_i_denom = bridge_waymark_grid[i + 1, :, i]

        ax.hist(bridge_i_numer, density=True, alpha=0.5, color='b',
                label="numerator", bins=int(len(bridge_i_numer) ** 0.5))
        ax.hist(bridge_i_denom, density=True, alpha=0.5, color='r',
                label="denominator", bins=int(len(bridge_i_denom) ** 0.5))

        ax.set_xlabel(r"$ \log r_{%s}(\mathbf{x}; \mathbf{\theta}_{%s})$" % (i, i))
        ax.legend()

    fig.tight_layout()
    save_fig(fig_dir, "{}_bridge_hists".format(which_set))


def plot_per_bridge_energy_graphs(neg_energy_grid, fig_dir, title, which_set, index_name="waymark"):

    neg_energy_grid, noise_dist_logp = neg_energy_grid[:, :, :-1], neg_energy_grid[:, :, -1]

    n = neg_energy_grid.shape[0]
    n_ratios = neg_energy_grid.shape[2]

    n_figs = n_ratios + 3
    n_cols = 4
    n_rows = int(np.ceil(n_figs/n_cols))

    set_all_fontsizes(6)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(1.5*n_cols, n_rows))
    fig.suptitle(title, fontsize=12)
    axs = axs.ravel()

    av_neg_energy_grid = np.mean(neg_energy_grid, axis=1)  # (n, n_bridges)
    cmap, norm = custom_cmap_with_zero_included(neg_energy_grid)
    im = axs[0].imshow(av_neg_energy_grid, norm=norm, cmap=cmap)
    axs[0].set_xlabel("bridge index")
    axs[0].set_ylabel("{} index".format(index_name))
    fig.colorbar(im, ax=axs[0])

    plotscatter_single_axis(axs[1], np.sum(av_neg_energy_grid, axis=1), "{} index".format(index_name), "total neg energy")

    for i in range(2, 2+n_ratios):
        plotscatter_single_axis(axs[i], av_neg_energy_grid[:, i-2],
            "{} index".format(index_name), "bridge {}".format(i-2))

    plotscatter_single_axis(axs[n_figs-1], np.mean(noise_dist_logp, axis=1), "{} index".format(index_name), "noise dist")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig_dir, "{}_all_{}s_all_bridges_energies".format(which_set, index_name))
    reset_all_fontsizes()

    return n


def compute_loss_wmark_i_vs_j_greater_than_i(loss_grid, sep_terms_loss_grid, bridge_waymark_grid, i):
    num_waymarks = bridge_waymark_grid.shape[0]
    for j in range(i+1, num_waymarks):
        class1 = np.sum(bridge_waymark_grid[i, :, i:j], axis=1)  # (sample_size, )
        class2 = np.sum(bridge_waymark_grid[j, :, i:j], axis=1)  # (sample_size, )
        loss, loss_term1, loss_term2 = np_nce_loss(class1, class2)

        loss_grid[i, j] = loss
        sep_terms_loss_grid[i, j] = loss_term1
        sep_terms_loss_grid[j, i] = loss_term2


def make_comparative_density_histogram(graph, sess, samples, config, save_dir, name):

    model_logps, flow_logp = tf_batched_operation(sess,
                                                  [graph.bridges_plus_noise_logp, graph.flow_log_prob],
                                                  len(samples),
                                                  config.n_batch // config.total_num_ratios,
                                                  data_pholder=graph.data,
                                                  data=samples)
    model_logp = np.sum(model_logps, axis=-1)

    fig, ax = plt.subplots(1, 1)
    plot_hist(model_logp, alpha=0.5, ax=ax, y_lim=[0, 0.02], label="{}_tre_model".format(name))
    plot_hist(flow_logp, alpha=0.5, ax=ax, y_lim=[0, 0.02], label="{}_resmademog_model".format(name))
    fig.legend()
    fig.suptitle(name + "_densities")
    save_fig(save_dir, name + "_comparative_densities")


def load_post_annealed_init_states(sess, graph, val_dp, config, logger):

    ais_save_dir = get_ais_dir()
    ais_chains_file = ais_save_dir + "ais_chains.npz"
    ais_final_step_size = config.init_post_annealed_step_size

    if os.path.isfile(ais_chains_file):
        # ais_final_step_size = config.get("ais_final_step_sizes", [ais_final_step_size])[-1]
        post_annealed_initial_states = np.load(ais_chains_file)["samples"][:, -1, ...]
        logger.info("loading AIS chains")
    else:
        post_annealed_initial_states = \
            sample_noise_dist(sess, graph, config.noise_dist_name, val_dp, config.only_sample_n_chains)
        logger.info("Using noise samples to initialise MCMC sampling from full model")

    return post_annealed_initial_states, ais_final_step_size


# noinspection PyUnresolvedReferences
def get_ais_dir(subdir=""):
    ais_dir_path = os.path.join(save_dir, "ais/{}/".format(ais_id))
    if subdir:
        ais_dir_path = os.path.join(ais_dir_path, subdir)
    os.makedirs(ais_dir_path, exist_ok=True)
    return ais_dir_path


def set_num_chains(config, logger):

    if config["dataset_name"] == "mnist":
        num_chains = max(int((20 / config.total_num_ratios) * 50), 50)
        if config["flow_type"] == "GLOW":
            num_chains = int(num_chains / 4)
    else:
        return

    logger.info("USING {} MCMC CHAINS".format(num_chains))
    config["ais_n_chains"] = num_chains
    config["only_sample_n_chains"] = num_chains


def make_global_config():
    """load & augment experiment configuration, then add it to global variables"""
    parser = ArgumentParser(description='Evaluate TRE model.', formatter_class=ArgumentDefaultsHelpFormatter)

    # parser.add_argument('--config_path', type=str, default="1d_gauss/20200501-0739_0")
    parser.add_argument('--config_path', type=str, default="gaussians/20200713-1029_4")
    # parser.add_argument('--config_path', type=str, default="mnist/20200504-1031_0")
    parser.add_argument('--ais_id', type=int, default=0)
    parser.add_argument('--eval_epoch_idx', type=str, default="best")

    parser.add_argument('--do_estimate_log_par', type=int, default=0)  # -1 == False, else True
    parser.add_argument('--do_sample', type=int, default=-1)  # -1 == False, else True
    parser.add_argument('--ais_nuts_max_tree_depth', type=int, default=5)  # -1 == False, else True

    parser.add_argument('--do_assess_subbridges', type=int, default=-1)  # -1 == False, else True
    parser.add_argument('--do_assess_parameters', type=int, default=0)  # -1 == False, else True

    parser.add_argument('--sample_method', type=str, default="nuts")
    parser.add_argument('--act_threshold_quantile', type=float, default=0.99)

    # if we are only sampling (i.e. not computing partition function with AIS), then this is the number of sampling
    # steps we use when performing annealed sampling. If None, then use the default value stored in config file.
    parser.add_argument('--only_sample_total_n_steps', type=int, default=1000)
    parser.add_argument('--only_sample_n_chains', type=int, default=-1)

    # initial step size for annealed sampling
    parser.add_argument('--ais_step_size_init', type=float, default=0.02)
    parser.add_argument('--init_post_annealed_step_size', type=float, default=0.02)

    # when doing annealed sampling with uncalibrated_langevin, we use an exponentially decreasing step size schedule.
    # The final step size in this schedule is 10^-step_size_reduction_magnitude smaller than the initial step size.
    parser.add_argument('--step_size_reduction_magnitude', type=float, default=2)

    # After annealed sampling, we continue sampling from the entire model
    parser.add_argument('--do_post_annealed_sample', type=int, default=0)  # -1 == False, else True
    parser.add_argument('--post_ais_n_samples_keep', type=int, default=20)
    parser.add_argument('--post_ais_thinning_factor', type=int, default=0)
    parser.add_argument('--post_ais_nuts_max_tree_depth', type=int, default=10)

    parser.add_argument('--parallel_iterations', type=int, default=10)
    parser.add_argument('--swap_memory', type=int, default=-1)  # attempt to save gpu memory by using cpu when possible

    parser.add_argument('--n_noise_samples_for_variational_losses', type=int, default=1000)
    parser.add_argument('--frac', type=float, default=1.0)
    parser.add_argument('--debug', type=int, default=-1)
    args = parser.parse_args()

    with open(project_root + "saved_models/{}/config.json".format(args.config_path)) as f:
        config = json.load(f)

    rename_save_dir(config)
    if args.only_sample_n_chains == -1:
        del args.only_sample_n_chains

    config.update(vars(args))

    config["do_estimate_log_par"] = False if args.do_estimate_log_par == -1 else True
    config["do_sample"] = False if args.do_sample == -1 else True
    config["do_post_annealed_sample"] = False if args.do_post_annealed_sample == -1 else True
    config["do_assess_subbridges"] = False if args.do_assess_subbridges == -1 else True
    config["do_assess_parameters"] = False if args.do_assess_parameters == -1 else True
    config["swap_memory"] = False if args.swap_memory == -1 else True
    config["debug"] = False if args.debug == -1 else True

    if config["eval_epoch_idx"] == "final":  # work out the final epoch number
        metrics_save_dir = os.path.join(config["save_dir"], "model/every_x_epochs/")
        epoch_nums = [x.split(".")[0] for x in os.listdir(metrics_save_dir) if "checkpoint" not in x]
        config["eval_epoch_idx"] = str(max([int(x) for x in epoch_nums]))

    if "data_dist_name" not in config: config["data_dist_name"] = None

    save_config(config)

    if config["debug"]:
        config["do_assess_subbridges"] = True
        config["do_assess_parameters"] = True
        config["do_sample"] = False
        config["do_estimate_log_par"] = True
        config["do_post_annealed_sample"] = False

        config["frac"] = 0.2

        config["ais_n_chains"] = 10
        config["ais_total_n_steps"] = 10
        config["only_sample_n_chains"] = 10
        config["only_sample_total_n_steps"] = 10
        config["post_ais_n_samples_keep"] = 10
        config["post_ais_thinning_factor"] = 5
        config["n_noise_samples_for_variational_losses"] = 1000

    globals().update(config)
    return AttrDict(config)


# noinspection PyUnresolvedReferences,PyTypeChecker
def main():
    """Assess a model learned with TRE using various metrics.

    These metrics can include:
    - plotting the neg-energy histograms of the model
    - evaluating the model under different loss functions
    - sampling from the model
    - estimating the log_partition function via AIS/RAISE
    """
    make_logger()
    logger = logging.getLogger("tf")
    np.set_printoptions(precision=4)

    # load a config file whose contents are added to globals(), making them easily accessible elsewhere
    config = make_global_config()

    val_dp = load_data_providers_and_update_conf(config, only_val=True)
    globals().update(config)
    ais_save_dir = get_ais_dir()

    # create a dictionary whose keys are tensorflow operations that can be accessed like attributes e.g graph.operation
    if config.noise_dist_name == "flow":
        graph = build_flow_based_graph(config)
    else:
        graph = build_graph(config)

    set_num_chains(config, logger)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        logger.info("Loading model from epoch: {}".format(config.eval_epoch_idx))
        load_model(sess, config.eval_epoch_idx, config)

        if config.noise_dist_name == "flow":
            load_flow(sess, config, config.flow_id)

        if config.do_assess_subbridges:
            assess_bridges(sess, graph, val_dp, config, which_set="val")

        if config.do_assess_parameters:
            assess_parameters(config)

        # perform annealed sampling from the model and maybe estimate log partition function with AIS/RAISE
        if config.do_sample or config.do_estimate_log_par:

            prenormalised_logp = evaluate_energies_and_losses(graph, sess, val_dp, ais_save_dir, config, logger)

            post_annealed_initial_states, ais_final_step_size = \
                run_annealing_methods(g=graph,
                                      sess=sess,
                                      val_dp=val_dp,
                                      prenormalised_logp=prenormalised_logp,
                                      ais_save_dir=ais_save_dir,
                                      config=config)

        # otherwise load final states from a previous annealed sample, or, if that's not possible,
        # initialise using samples from the noise distribution
        elif config.do_post_annealed_sample:
            post_annealed_initial_states, ais_final_step_size = \
                load_post_annealed_init_states(sess, graph, val_dp, config, logger)

        # continue running the MCMC sampler from the full model
        if config.do_post_annealed_sample:
            run_full_model_samplers(sess, graph, post_annealed_initial_states,
                                    ais_final_step_size, val_dp, ais_save_dir, config, logger)

        logger.info("Finished!")
        save_config(config)
        with open(os.path.join(ais_save_dir, "finished.txt"), 'w+') as f:
            f.write("finished.")


if __name__ == "__main__":
    main()
