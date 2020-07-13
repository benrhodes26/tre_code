from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import tensorflow_probability as tfp

from models import SeparableEnergy
from experiment_ops import build_energies, plot_chains, load_model
from keras_layers import ParallelDense
from tensorflow.keras.models import Model
from utils.experiment_utils import *
from utils.misc_utils import *
from utils.plot_utils import *
from utils.tf_utils import *

tfb = tfp.bijectors
tfd = tfp.distributions

# noinspection PyUnresolvedReferences
def build_placeholders(hidden_size, num_classification_problems):

    if "img_shape" in data_args and data_args["img_shape"] is not None:
        shp = data_args["img_shape"][:-1]
        data = tf.compat.v1.placeholder(tf.float32, (None, *shp), "data")
    else:
        raise NotImplementedError

    hidden_vectors = tf.compat.v1.placeholder(tf.float32, (None, hidden_size), "hidden_vectors")
    true_labels = tf.compat.v1.placeholder(tf.float32, (None, num_classification_problems), "true_labels")

    return AttrDict(locals())


def build_graph(config):

    pholders = build_placeholders(config.mlp_hidden_size, config.num_classification_problems)
    labels = tf.cast(pholders.true_labels, dtype=tf.int32)
    one_hot_labels = tf.one_hot(labels, depth=config.max_num_classes)

    with tf.variable_scope("tre_model"):
        energy_obj = SeparableEnergy(bridge_idxs=None, max_num_ratios=None, config=config, only_f=True)
        f_x = energy_obj.compute_f_hiddens(pholders.data, is_train=False)

    with tf.variable_scope("linear_probes"):
        inputs = k_layers.Input(shape=config.mlp_hidden_size)
        log_probs = ParallelDense(n_parallel=config.num_classification_problems,
                                  in_dim=config.mlp_hidden_size,
                                  out_dim=config.max_num_classes,
                                  activation=lambda x: tf.nn.log_softmax(x, axis=-1),
                                  )(inputs)
        model = Model(inputs=inputs, outputs=log_probs)

    # compute the accuracy
    model_output = model(pholders.hidden_vectors)
    accuracy = accuracy_fn(model_output, labels, config)
    loss = nll_loss(model_output, one_hot_labels, config)

    func = tf_lbfgs_function_factory(model=model,
                                     loss=lambda x: nll_loss(x, one_hot_labels=one_hot_labels, config=config),
                                     train_x=pholders.hidden_vectors,
                                     trace_fn=lambda x: accuracy_fn(x, true_labels=labels, config=config),
                                     trace_label="accuracy"
                                     )

    # convert initial model parameters to a 1D tf.Tensor
    init_params = tf.dynamic_stitch(func.idx, model.trainable_variables)

    # train the model with L-BFGS solver
    # 'lbfgs_results' is a named tuple, with the following keys of interest:
    # "converged", "num_objective_evaluations", "position", "objective_value"
    lbfgs_results = tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=func,
        initial_position=init_params,
        f_relative_tolerance=1e-5,
        max_iterations=10000
    )

    # after training, the final optimized parameters are still in lbfgs_results.position
    # so we have to manually put them back to the model
    func.assign_new_model_parameters(lbfgs_results.position)

    converged = lbfgs_results.converged
    n_func_evals = lbfgs_results.num_objective_evaluations
    final_loss = lbfgs_results.objective_value

    graph = AttrDict(locals())
    graph.update(pholders)
    return graph  # dict whose values can be accessed as attributes i.e. val = dict.key


def nll_loss(log_probs, one_hot_labels, config):

    masked_log_probs = tf.identity(log_probs)
    if config.num_classification_problems > 1:
        masked_log_probs = mask_extra_dims(config, masked_log_probs, zero_mask_output=True)

    log_likelihood = tf.reduce_sum(one_hot_labels * masked_log_probs, axis=-1)
    av_nll = -tf.reduce_mean(log_likelihood)

    return av_nll


def accuracy_fn(log_probs, true_labels, config):

    masked_log_probs = tf.identity(log_probs)
    if config.num_classification_problems > 1:
        masked_log_probs = mask_extra_dims(config, masked_log_probs, zero_mask_output=False)

    predictions = tf.argmax(masked_log_probs, axis=-1)  # (n, num_classification_problems)
    predictions = tf.cast(predictions, dtype=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, true_labels), dtype=tf.float32))

    return accuracy


def mask_extra_dims(config, log_probs, zero_mask_output):
    """For each classification problem, we are outputting the same number of logits, despite the fact
    that different problems have different number of classes. Thus, for each problem,
    we mask out unnecessary logits (and renormalize). This leads to (increased) over-parameterisation
    but given that the usual softmax cross-entropy loss is overparameterised to start with, it's
    probably not a big deal.
    """

    class_mask = np.zeros((config.num_classification_problems, config.max_num_classes), dtype=np.float32)
    neg_infs = np.zeros_like(class_mask)

    num_classes_per_problem = config["num_classes_per_problem"]
    for i in range(config.num_classification_problems):
        class_mask[i, :num_classes_per_problem[i]] = 1.0
        neg_infs[i, num_classes_per_problem[i]:] = -np.inf

    class_mask = tf.convert_to_tensor(class_mask, dtype=tf.float32)  # (num_classification_problems, max_num_classes)
    neg_infs = tf.convert_to_tensor(neg_infs)

    # renormalise the non-masked entries
    masked_log_probs = (class_mask * log_probs) + neg_infs  # (n, num_classification_problems, max_num_classes)
    masked_log_probs -= tf.reduce_logsumexp(masked_log_probs, axis=-1, keepdims=True) * class_mask

    if zero_mask_output:
        masked_log_probs = tf.where(tf.is_inf(masked_log_probs), tf.zeros_like(masked_log_probs), masked_log_probs)

    return masked_log_probs


def train_linear_probes(g, sess, dps, config):

    logger = logging.getLogger("tf")
    train_dp, val_dp, test_dp = dps
    eval_save_dir = get_save_dir(config.save_dir)

    # feed all data through f to compute the hidden representations
    per_set_hiddens = []
    for dp in dps:
        data = dp.data[..., 0]
        h_vecs = tf_batched_operation(sess=sess,
                                      ops=[g.f_x],
                                      n_samples=len(dp.data),
                                      batch_size=config.n_batch,
                                      data_pholder=g.data,
                                      data=data)
        per_set_hiddens.append(h_vecs)

    logger.info("Intial train loss: {}".format(
          sess.run(g.loss, feed_dict={g.hidden_vectors: per_set_hiddens[0], g.true_labels: train_dp.labels}))
    )

    # feed all representations into graph and optimise via LBFGS
    res = sess.run([g.converged, g.n_func_evals, g.final_loss],
                   feed_dict={
                       g.hidden_vectors: per_set_hiddens[0],  # train set hiddens
                       g.true_labels: train_dp.labels
                   }
                   )

    logger.info("Finished training!")
    logger.info("Converged status: {}".format(res[0]))
    logger.info("Num function evals: {}".format(res[1]))
    logger.info("Final loss: {}".format(res[2]))

    # compute accuracy for each set
    accs = []
    set_names = ["train", "val", "test"]
    for h, dp, set_name in zip(per_set_hiddens, dps, set_names):
        acc = sess.run(g.accuracy, feed_dict={g.hidden_vectors: h, g.true_labels: dp.labels})
        logger.info("{} final accuracy: {}".format(set_name, acc))
        accs.append(acc)

    config["representation_learning_train_acc"] = accs[0]
    config["representation_learning_val_acc"] = accs[1]
    config["representation_learning_test_acc"] = accs[-1]
    np.savez(
        path_join(eval_save_dir, "final_loss_accuracy"),
        loss=np.array([res[2]]),
        accs=np.array(accs)
    )


# noinspection PyUnresolvedReferences
def get_save_dir(save_dir, subdir=""):
    eval_save_dir = os.path.join(save_dir, "repr_eval/")
    if subdir: eval_save_dir = os.path.join(eval_save_dir, subdir)
    os.makedirs(eval_save_dir, exist_ok=True)
    return eval_save_dir


def make_global_config():
    """load & augment experiment configuration, then add it to global variables"""
    parser = ArgumentParser(description='Evaluate TRE model using linear probes trained on fixed represenatations'
                            , formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config_path', type=str, default="multiomniglot/20200511-0818_0")
    parser.add_argument('--eval_epoch_idx', type=str, default="best")
    parser.add_argument('--frac', type=float, default=1.0)
    parser.add_argument('--debug', type=int, default=-1)
    args = parser.parse_args()


    with open(project_root + "saved_models/{}/config.json".format(args.config_path)) as f:
        config = json.load(f)

    rename_save_dir(config)
    config.update(vars(args))

    config["debug"] = False if args.debug == -1 else True

    if config["eval_epoch_idx"] == "final":  # work out the final epoch number
        metrics_save_dir = os.path.join(config["save_dir"], "model/every_x_epochs/")
        epoch_nums = [x.split(".")[0] for x in os.listdir(metrics_save_dir) if "checkpoint" not in x]
        config["eval_epoch_idx"] = str(max([int(x) for x in epoch_nums]))

    metrics_save_dir = path_join(config["save_dir"], "metrics/epoch_{}".format(config["eval_epoch_idx"]))
    config["prenormalised_kl"] = np.sum(np.load(path_join(metrics_save_dir, "val.npz"))["energy"])
    config["total_num_ratios"] = int(config["data_args"]["n_imgs"] / config["waymark_mixing_increment"])

    save_config(config)
    if config["debug"]:
        config["frac"] = 0.1

    globals().update(config)
    return AttrDict(config)


# noinspection PyUnresolvedReferences,PyTypeChecker
def main():
    """Assess a model learned with TRE via linear probes built on top of fixed representations
    """
    make_logger()
    logger = logging.getLogger("tf")
    np.set_printoptions(precision=4)

    # load a config file whose contents are added to globals(), making them easily accessible elsewhere
    config = make_global_config()

    dps = load_data_providers_and_update_conf(config, include_test=True, use_labels=True)

    # create a dictionary whose keys are tensorflow operations that can be accessed like attributes e.g graph.operation
    graph = build_graph(config)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        logger.info("Loading model from epoch: {}".format(config.eval_epoch_idx))
        load_model(sess, config.eval_epoch_idx, config)

        probes = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='linear_probes/')
        sess.run(tf.compat.v1.variables_initializer(probes))
        train_linear_probes(graph, sess, dps, config)

        logger.info("Finished!")
        eval_save_dir = get_save_dir(config.save_dir)
        save_config(config)
        with open(os.path.join(eval_save_dir, "finished.txt"), 'w+') as f:
            f.write("finished.")


if __name__ == "__main__":
    main()
