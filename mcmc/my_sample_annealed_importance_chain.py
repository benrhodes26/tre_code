# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Markov chain Monte Carlo driver, `sample_chain_annealed_importance_chain`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
# Dependency imports
import numpy as np

import tensorflow as tf
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

__all__ = [
    "sample_annealed_importance_chain",
    "sample_annealed_chain"
]

AISResults = collections.namedtuple(
    "AISResults",
    [
        "target_log_prob",
        "inner_results",
        "step_size",
    ])


def sample_annealed_importance_chain(
        num_steps,
        all_energy_fns,
        target_energy_fn,
        current_state,
        make_kernel_fn,
        init_step_size,
        kernel_results=None,
        forward=True,
        do_compute_ais_weights=True,
        initial_weights=None,
        has_accepted_results=True,
        parallel_iterations=10,
        swap_memory=False,
        name=None):
    """Runs annealed importance sampling (AIS) to estimate normalizing constants for TRE model

  Note: `proposal_log_prob_fn` and `target_log_prob_fn` are called exactly three
  times (although this may be reduced to two times, in the future).

  Args:
    num_steps: Integer number of Markov chain updates to run. More
      iterations means more expense, but smoother annealing between q
      and p, which in turn means exponentially lower variance for the
      normalizing constant estimator.

    all_energy_fns: list of Python callables that returns the log-density/energy of the
      noise distribution and all bridges up to (and including) the current target bridge.
      Each element of the list is a function that computes all bridge energies for a given subnetwork
      and thus each functions output has shape [n_chains, n_ratios_subnet_i].

    target_energy_fn: Python callable which returns the log-density/energy under the target bridge.

    current_state: `Tensor` or Python `list` of `Tensor`s representing the
      current state(s) of the Markov chain(s). The first `r` dimensions index
      independent chains, `r = tf.rank(target_log_prob_fn(*current_state))`.

    make_kernel_fn: Python `callable` which returns a `TransitionKernel`-like
      object. Must take one argument representing the `TransitionKernel`'s
      `target_log_prob_fn`. The `target_log_prob_fn` argument represents the
      `TransitionKernel`'s target log distribution.  Note:
      `sample_annealed_importance_chain` creates a new `target_log_prob_fn`
      which
    is an interpolation between the supplied `target_log_prob_fn` and
    `proposal_log_prob_fn`; it is this interpolated function which is used as an
    argument to `make_kernel_fn`.

    parallel_iterations: The number of iterations allowed to run in parallel.
        It must be a positive integer. See `tf.while_loop` for more details.

    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., "sample_annealed_importance_chain").

  Returns:
    next_state: `Tensor` or Python list of `Tensor`s representing the
      state(s) of the Markov chain(s) at the final iteration. Has same shape as
      input `current_state`.
    ais_weights: Tensor with the estimated weight(s). Has shape matching
      `target_log_prob_fn(current_state)`.
    kernel_results: `collections.namedtuple` of internal calculations used to
      advance the chain.
  """

    # todo: simplify. This code is unnecessarily complex since it was designed to handle multiple networks, each of which would
    # compute a subset of the bridges used in TRE.

    with tf.name_scope(name, "sample_annealed_importance_chain", [num_steps, current_state]):
        num_steps = tf.convert_to_tensor(num_steps, dtype=tf.int32, name="num_steps")
        if mcmc_util.is_list_like(current_state):
            current_state = [tf.convert_to_tensor(s, name="current_state") for s in current_state]
        else:
            current_state = tf.convert_to_tensor(current_state, name="current_state")

        def _make_convex_combined_log_prob_fn(iter_):
            def _fn(*args):
                all = [f(*args) for f in all_energy_fns]

                prev_energies_in_cur_subnet = all[-1][:, 1:]
                if prev_energies_in_cur_subnet.get_shape().as_list()[1] > 0:
                    prev = all[:-1] + [prev_energies_in_cur_subnet]
                else:
                    prev = all[:-1]

                prev_energies = tf.add_n([tf.reduce_sum(e, axis=-1) for e in prev])
                next_energies = all[-1][:, 0]

                p = tf.identity(prev_energies, name="previous_energies")
                t = tf.identity(next_energies, name="target_log_prob")
                dtype = p.dtype.base_dtype
                if forward:
                    beta = tf.cast(iter_ + 1, dtype) / tf.cast(num_steps, dtype)
                else:
                    beta = 1. - (tf.cast(iter_ + 1, dtype) / tf.cast(num_steps, dtype))

                return tf.identity(p + (beta * t), name="convex_combined_log_prob")

            return _fn

        def _loop_body(iter_, ais_weights, current_state, kernel_results):
            """Closure which implements `tf.while_loop` body."""
            x = (current_state if mcmc_util.is_list_like(current_state) else [current_state])

            if not do_compute_ais_weights:
                target_log_prob = kernel_results.target_log_prob
            else:
                target_log_prob = target_energy_fn(*x)
                if forward:
                    ais_weights += (target_log_prob / tf.cast(num_steps, ais_weights.dtype))
                else:
                    ais_weights -= (target_log_prob / tf.cast(num_steps, ais_weights.dtype))

            kernel = make_kernel_fn(_make_convex_combined_log_prob_fn(iter_), kernel_results.step_size)
            next_state, inner_results = kernel.one_step(current_state, kernel_results.inner_results)

            if hasattr(inner_results.inner_results, "accepted_results"):
                step_size = inner_results.inner_results.accepted_results.step_size
            else:
                step_size = inner_results.inner_results.step_size

            kernel_results = AISResults(target_log_prob=target_log_prob, inner_results=inner_results, step_size=step_size)

            return [iter_ + 1, ais_weights, next_state, kernel_results]

        def _bootstrap_results(init_state):
            """Creates first version of `kernel_results`."""
            kernel = make_kernel_fn(_make_convex_combined_log_prob_fn(iter_=0), init_step_size)
            inner_results = kernel.bootstrap_results(init_state)

            if has_accepted_results:
                convex_combined_log_prob = inner_results.inner_results.accepted_results.target_log_prob
            else:
                convex_combined_log_prob = inner_results.inner_results.target_log_prob
            dtype = convex_combined_log_prob.dtype.as_numpy_dtype
            shape = tf.shape(convex_combined_log_prob)
            target_log_prob = tf.fill(shape, dtype(np.nan), name="target_target_log_prob")

            if hasattr(inner_results.inner_results, "accepted_results"):
                step_size = inner_results.inner_results.accepted_results.step_size
            else:
                step_size = inner_results.inner_results.step_size

            return AISResults(target_log_prob=target_log_prob, inner_results=inner_results, step_size=step_size)

        if kernel_results is None:
            kernel_results = _bootstrap_results(current_state)
        inner_results = kernel_results.inner_results

        if initial_weights is None:
            if has_accepted_results:
                ais_weights = tf.zeros(
                    shape=tf.broadcast_dynamic_shape(
                        tf.shape(inner_results.inner_results.proposed_results.target_log_prob),
                        tf.shape(inner_results.inner_results.accepted_results.target_log_prob)),
                    dtype=inner_results.inner_results.proposed_results.target_log_prob.dtype.base_dtype)
            else:
                ais_weights = tf.zeros(
                    shape=tf.shape(inner_results.inner_results.target_log_prob),
                    dtype=inner_results.inner_results.target_log_prob.dtype.base_dtype)
        else:
            ais_weights = initial_weights

        [_, ais_weights, current_state, kernel_results] = tf.while_loop(
            cond=lambda iter_, *args: iter_ < num_steps,
            body=_loop_body,
            loop_vars=[
                np.int32(0),  # iter_
                ais_weights,
                current_state,
                kernel_results,
            ],
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory)

        return [current_state, ais_weights, kernel_results]
