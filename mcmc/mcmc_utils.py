import tensorflow_probability as tfp
tfd = tfp.distributions

from mcmc.my_hmc import HamiltonianMonteCarlo
from mcmc.my_nuts import NoUTurnSampler as NoUTurnSampler_v1
from mcmc.my_nuts_v2 import NoUTurnSampler as NoUTurnSampler_v2
from mcmc.my_sample import sample_chain
from experiment_ops import build_noise_dist
from utils.misc_utils import *
from utils.tf_utils import *
from utils.experiment_utils import *
from utils.plot_utils import *


# noinspection PyUnresolvedReferences
def build_mcmc_chain(target_log_prob_fn,
                     initial_states,
                     n_samples_to_keep,
                     thinning_factor,
                     mcmc_method,
                     step_size,
                     use_adaptive_step_size=True,
                     n_adaptation_steps=0,
                     n_leapfrog_steps=10,
                     nuts_max_tree_depth=10,
                     parallel_iterations=10,
                     swap_memory=False):

    use_mh_step = True
    if mcmc_method == "nuts":
        # kernel = tfp.mcmc.NoUTurnSampler(
        kernel = NoUTurnSampler_v2(
            target_log_prob_fn=target_log_prob_fn,
            max_tree_depth=nuts_max_tree_depth,
            step_size=step_size,
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory
        )

        if use_adaptive_step_size:
            kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                kernel,
                target_accept_prob=0.6,
                num_adaptation_steps=n_adaptation_steps,
                step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(step_size=new_step_size),
                step_size_getter_fn=lambda pkr: pkr.step_size,
                log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio
            )

    elif mcmc_method == "hmc":
        # kernel = tfp.mcmc.HamiltonianMonteCarlo(
        kernel = HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=n_leapfrog_steps,
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory
        )

    elif mcmc_method == "metropolis_langevin":
        kernel = tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size
        )

    elif mcmc_method == "uncalibrated_metropolis_langevin":
        use_mh_step = False
        kernel = tfp.mcmc.UncalibratedLangevin(
                    target_log_prob_fn=target_log_prob_fn,
                    step_size=step_size,
                    compute_acceptance=False)

    elif mcmc_method == "random_walk":
        proposal = tfp.mcmc.random_walk_normal_fn(scale=step_size)
        kernel = tfp.mcmc.RandomWalkMetropolis(target_log_prob_fn, new_state_fn=proposal)

    else:
        raise ValueError("must specify a valid mcmc method. `{}' is not a valid choice.".format(mcmc_method))

    if use_adaptive_step_size and mcmc_method in ["hmc", "metropolis_langevin", "random_walk"]:
        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            kernel, num_adaptation_steps=n_adaptation_steps, target_accept_prob=0.6)

    # chains, kernel_results = tfp.mcmc.sample_chain(
    chains, kernel_results = sample_chain(
        num_results=n_samples_to_keep,
        current_state=initial_states,
        kernel=kernel,
        num_burnin_steps=thinning_factor,
        num_steps_between_results=thinning_factor,
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory)

    if use_mh_step:
        res = kernel_results.inner_results
        log_n = tf.math.log(tf.cast(tf.size(res.log_accept_ratio), res.log_accept_ratio.dtype))
        log_mean_accept_ratio = tf.reduce_logsumexp(tf.minimum(res.log_accept_ratio, 0.)) - log_n
        accept_rate = tf.exp(log_mean_accept_ratio)
    else:
        accept_rate = tf.constant(1.0, tf.float32)

    if mcmc_method == "nuts":
        nuts_leapfrogs_taken = tf.reduce_mean(kernel_results.inner_results.leapfrogs_taken, axis=1)
    else:
        nuts_leapfrogs_taken = tf.no_op()

    if hasattr(kernel_results, "inner_results"):
        if hasattr(kernel_results.inner_results, "accepted_results"):
            ss = kernel_results.inner_results.accepted_results.step_size
        else:
            ss = kernel_results.inner_results.step_size
    else:
        ss = step_size  # step size hasn't been adapted

    # tranpose first two axes
    event_dims = shape_list(chains)[2:]
    new_order = [1, 0] + [i for i in range(2, 2 + len(event_dims))]
    chains = tf.transpose(chains, new_order)  # (n_chains, chain_length, ...)

    return chains, accept_rate, ss, nuts_leapfrogs_taken


# noinspection PyUnresolvedReferences
def build_mcmc_graph(config):
    """Build graph to sample from MCMC chain for TRE

    Returns: dictionary of of all local variables, including the graph ops required for training
    """
    if (config.data_args is not None) and ("img_shape" in config.data_args):
        initial_states = tf.placeholder(tf.float32, (None, *config.data_args["img_shape"]), "mcmc_initial_states")
    else:
        initial_states = tf.placeholder(tf.float32, (None, config.n_dims), "mcmc_initial_states")
    n_samples_to_keep = tf.placeholder(tf.int32, (), "n_samples_to_keep")
    thinning_factor = tf.placeholder(tf.int64, (), "mcmc_thinning_factor")
    step_size = tf.placeholder_with_default(0.0, (), "step_size")

    d_shape = shape_list(initial_states)[1:]  # shape of non_batch dims

    raise NotImplementedError
    # _, mcmc_noise_dist = build_noise_dist(config.noise_dist_name, initial_states, config, d_shape)

    sequence_data, mcmc_accept_rate, _, _ = build_mcmc_chain(target_log_prob_fn=mcmc_noise_dist.log_prob,
                                                             initial_states=initial_states,
                                                             n_samples_to_keep=n_samples_to_keep,
                                                             thinning_factor=thinning_factor,
                                                             mcmc_method=config.mcmc_method,
                                                             step_size=step_size)

    return AttrDict(locals())  # dict whose values can be accessed as attributes i.e. val = dict.key


# noinspection PyUnresolvedReferences
def create_and_save_mcmc_chain(graph, sess, noise_samples, dp, dir_path, config, time_step=None):
    # Track the time that it takes to sample
    pre_sample_time = time()
    data_shape = dp.data.shape
    n = data_shape[0]
    batch_size = min(1000, n)  # size of batch for sampling

    # work out number of waymarks we are going to generate
    max_number_of_ratios = 1
    for param in config.mcmc_waymark_params:
        max_number_of_ratios += param[-1]

    # save one batch of complete chains, for plotting
    plotting_data = np.zeros((batch_size, 1 + max_number_of_ratios) + data_shape[1:])

    # For each datapoint, generate an mcmc chain that converges to a noise distribution.
    # For memory reasons, do this in stages, generating sub-chains in a loop.
    initial_states = dp.data
    mcmc_accept_rates = []
    time_step = 1 if time_step is None else time_step
    first_time_step = time_step

    all_step_sizes = []
    for params in config.mcmc_waymark_params:
        start_step, end_step, num_intervals = params
        all_step_sizes.append(np.linspace(start_step, end_step, num_intervals))

    step_sizes = np.concatenate(all_step_sizes)
    for step_size in step_sizes[time_step-1:]:

        seq_data, mcmc_accept_rate = sample_sub_chain(g=graph,
                                                      sess=sess,
                                                      initial_states=initial_states,
                                                      batch_size=batch_size,
                                                      n_samples_to_keep=1,
                                                      thinning_factor=1,
                                                      step_size=step_size,
                                                      data_shape=data_shape
                                                      )
        mcmc_accept_rates.append(mcmc_accept_rate)

        for j in range(time_step, time_step + seq_data.shape[1]):
            np.savez_compressed(dir_path + "waymark_data_{}".format(j), data=seq_data[:, j-time_step, ...])

        # save a batch of data for plotting
        subchain_len = seq_data.shape[1]
        plotting_data[:, time_step:time_step+subchain_len, ...] = seq_data[:batch_size, ...]

        time_step += seq_data.shape[1]
        initial_states = seq_data[:, -1, ...]

    # save initial and final samples
    plotting_data[:, first_time_step-1, ...] = dp.data[:batch_size]
    plotting_data[:, time_step, ...] = noise_samples[:batch_size]
    np.savez_compressed(dir_path + "waymark_data_{}".format(first_time_step-1), data=dp.data)
    np.savez_compressed(dir_path + "waymark_data_{}".format(time_step), data=noise_samples)

    # save how long it took
    update_time_avg(config, pre_sample_time, "mean_sequence_sample_time", "sequence_sampling_counter")

    return plotting_data, np.array(mcmc_accept_rates)


def sample_sub_chain(g,
                     sess,
                     initial_states,
                     batch_size,
                     n_samples_to_keep,
                     thinning_factor,
                     step_size,
                     data_shape,
                     ):
    n = data_shape[0]
    accept_rates = []

    # generate subchains in minibatches
    seq_data = np.zeros((data_shape[0], n_samples_to_keep) + data_shape[1:])
    for j in range(0, n, batch_size):
        fd = {g.n_samples_to_keep: n_samples_to_keep,
              g.thinning_factor: thinning_factor,
              g.step_size: step_size}
        ops = [g.sequence_data, g.mcmc_accept_rate]

        if j + batch_size < n:
            fd.update({g.initial_states: initial_states[j:j + batch_size]})
            seq_data[j:j + batch_size, ...], ar = sess.run(ops, feed_dict=fd)
        else:
            fd.update({g.initial_states: initial_states[j:]})
            seq_data[j:, ...], ar = sess.run(ops, feed_dict=fd)

        accept_rates.append(ar)

    return seq_data, np.mean(np.array(accept_rates))
