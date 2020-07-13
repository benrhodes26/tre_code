import tensorflow_probability as tfp
tfd = tfp.distributions

from utils.tf_utils import *

class LogisticLoss:

    def __init__(self, nu, label_smoothing_alpha=0.0, one_sided_smoothing=True):
        self.nu = nu
        self.log_nu = tf.log(nu)
        self.class1_acc = None
        self.class2_acc = None
        self.acc = None
        self.dawid_statistic_numerator = None
        self.dawid_statistic_denominator = None
        assert 0 <= label_smoothing_alpha < 0.5, "label smoothing parameter should be between 0 & 0.5"
        self.ls_alpha = label_smoothing_alpha
        self.one_sided_smoothing = one_sided_smoothing

    @tf_var_scope
    def loss(self, neg_energy):
        """Returns average over K Logistic losses given negative energies of model

        Args:
            neg_energy:  (2 * n_batch, n_ratios)
        """
        neg_energy1, neg_energy2 = tf.split(neg_energy, num_or_size_splits=2, axis=0)  # each (n, n_losses)

        term1 = tf.log_sigmoid(neg_energy1 - self.log_nu)  # (n, n_losses)
        term2 = tf.log_sigmoid(self.log_nu - neg_energy2)  # (n, n_losses)

        if self.ls_alpha > 0:
            term1, term2 = self.apply_label_smoothing(neg_energy1, neg_energy2, term1, term2)

        self.compute_classification_acc(term1, term2)
        self.compute_dawid_statistic(term1, term2)

        loss = -tf.reduce_mean(term1, axis=0) - self.nu * tf.reduce_mean(term2, axis=0)  # (n_losses, )

        return loss, -term1, -self.nu*term2

    def apply_label_smoothing(self, neg_energy1, neg_energy2, term1, term2):

        term1 *= (1 - self.ls_alpha)
        term1 += self.ls_alpha * tf.log_sigmoid(-neg_energy1)
        if not self.one_sided_smoothing:
            term2 *= (1 - self.ls_alpha)
            term2 += self.ls_alpha * tf.log_sigmoid(neg_energy2)

        return term1, term2

    def compute_classification_acc(self, term1, term2):

        class1_scores = tf.where(term1 > -tf.log(2.), tf.ones_like(term1, dtype=tf.float32),
                                 tf.zeros_like(term1, dtype=tf.float32))
        class2_scores = tf.where(term2 > -tf.log(2.), tf.ones_like(term2, dtype=tf.float32),
                                 tf.zeros_like(term2, dtype=tf.float32))

        self.class1_acc = tf.reduce_mean(class1_scores, axis=0)  # (n_losses,)
        self.class2_acc = tf.reduce_mean(class2_scores, axis=0)  # (n_losses,)

        self.acc = 0.5 * (self.class1_acc + self.class2_acc)  # (n_losses,)

    def compute_dawid_statistic(self, term1, term2):
        """Dawid, A. P. Prequential analysis. Encyclopedia of Statistical Sciences, 1:464–470, 1997"""
        num_class1 = tf.cast(shape_list(term1)[0], tf.float32)

        class1_p1 = tf.exp(term1)
        class2_p1 = 1 - tf.exp(term2)

        sum_class1_p1 = tf.reduce_sum(class1_p1, axis=0)  # (n_losses)
        sum_class2_p1 = tf.reduce_sum(class2_p1, axis=0)  # (n_losses)

        class1_bernoulli_var = tf.reduce_sum(class1_p1 * (1 - class1_p1), axis=0)  # (n_losses)
        class2_bernoulli_var = tf.reduce_sum(class2_p1 * (1 - class2_p1), axis=0)  # (n_losses)

        self.dawid_statistic_numerator = num_class1 - sum_class1_p1 - sum_class2_p1  # (n_losses)
        self.dawid_statistic_denominator = class1_bernoulli_var + class2_bernoulli_var  # (n_losses)


class DVLoss:

    def __init__(self):
        self.term1 = None
        self.term2 = None

    @tf_var_scope
    def loss(self, neg_energy):
        """Returns Donsker-Varadhan loss

        Args:
            neg_energy:  (2 * n_batch,) - a.k.a the log-density ratio
        """
        neg_energy1, neg_energy2 = tf.split(neg_energy, num_or_size_splits=2, axis=0)  # each (n_ratios,)
        n_samples = shape_list(neg_energy2)

        self.term1 = -tf.reduce_mean(neg_energy1)
        self.term2 = tf.reduce_logsumexp(neg_energy2) - tf.log(n_samples)

        return self.term1 + self.term2


class NWJLoss:

    def __init__(self):
        self.term1 = None
        self.term2 = None
        self.acc = None
        self.class1_acc = None
        self.class2_acc = None
        self.dawid_statistic_numerator = None
        self.dawid_statistic_denominator = None

    @tf_var_scope
    def loss(self, neg_energy):
        """Returns Donsker-Varadhan loss

        Args:
            neg_energy:  (2 * n_batch,) -
        """
        neg_energy1, neg_energy2 = tf.split(neg_energy, num_or_size_splits=2, axis=0)  # each (n_ratios,)

        class1_scores = tf.where(neg_energy1 > 0.0, tf.ones_like(neg_energy1, dtype=tf.float32),
                                 tf.zeros_like(neg_energy1, dtype=tf.float32))
        class2_scores = tf.where(neg_energy2 < 0.0, tf.ones_like(neg_energy2, dtype=tf.float32),
                                 tf.zeros_like(neg_energy2, dtype=tf.float32))

        self.class1_acc = tf.reduce_mean(class1_scores, axis=0)  # (n_ratios,)
        self.class2_acc = tf.reduce_mean(class2_scores, axis=0)  # (n_ratios,)
        self.dawid_statistic_numerator = tf.zeros_like(self.class1_acc)
        self.dawid_statistic_denominator = tf.zeros_like(self.class1_acc)
        self.acc = 0.5 * (self.class1_acc + self.class2_acc)  # (n_ratios,)

        self.term1 = -tf.reduce_mean(neg_energy1, axis=0) - 1
        self.term2 = tf.reduce_mean(tf.exp(neg_energy2), axis=0)


        return self.term1 + self.term2, -neg_energy1-1, tf.exp(neg_energy2)


class LSQLoss:

    def __init__(self):
        self.term1 = None
        self.term2 = None
        self.acc = None
        self.class1_acc = None
        self.class2_acc = None
        self.dawid_statistic_numerator = None
        self.dawid_statistic_denominator = None

    @tf_var_scope
    def loss(self, neg_energy):
        """Returns least-square losss

        Args:
            neg_energy:  (2 * n_batch,) -
        """
        neg_energy1, neg_energy2 = tf.split(neg_energy, num_or_size_splits=2, axis=0)  # each (n_ratios,)

        class1_scores = tf.where(neg_energy1 > 0.0, tf.ones_like(neg_energy1, dtype=tf.float32),
                                 tf.zeros_like(neg_energy1, dtype=tf.float32))
        class2_scores = tf.where(neg_energy2 < 0.0, tf.ones_like(neg_energy2, dtype=tf.float32),
                                 tf.zeros_like(neg_energy2, dtype=tf.float32))

        self.class1_acc = tf.reduce_mean(class1_scores, axis=0)  # (n_ratios,)
        self.class2_acc = tf.reduce_mean(class2_scores, axis=0)  # (n_ratios,)
        self.dawid_statistic_numerator = tf.zeros_like(self.class1_acc)
        self.dawid_statistic_denominator = tf.zeros_like(self.class1_acc)
        self.acc = 0.5 * (self.class1_acc + self.class2_acc)  # (n_ratios,)

        term1 = 0.5 * (tf.sigmoid(neg_energy1) - 1)**2
        term2 = 0.5 * tf.sigmoid(neg_energy2)**2

        loss = tf.reduce_mean(term1, axis=0) + tf.reduce_mean(term2, axis=0)

        return loss, term1, term2


class CNCELoss:

    def __init__(self, cnce_loss, model_type):
        assert model_type == "inverted", "This loss function has currently only be designed" \
                                         "to work for an 'inverted' model."
        self.cnce_loss = cnce_loss
        self.c = 0
        self.data_acc = -1
        self.noise_acc = -1  # ill-defined for CNCE
        self.acc = -1  # ill-defined for CNCE

    @tf_var_scope
    def loss(self, condprob_data_given_noise, condprob_noise_given_data, ebm_log_p_data, ebm_log_p_noise):
        if self.cnce_loss == "symmetric":
            log_class_prob = tf.nn.softplus(ebm_log_p_noise - ebm_log_p_data)
        else:
            log_class_prob = tf.nn.softplus(ebm_log_p_noise + condprob_data_given_noise
                                            - ebm_log_p_data - condprob_noise_given_data)

        self.acc = tf.reduce_mean(tf.where(log_class_prob < tf.log(2.),
                                           tf.ones_like(log_class_prob, dtype=tf.float32),
                                           tf.zeros_like(log_class_prob, dtype=tf.float32)))

        return 2 * tf.reduce_mean(log_class_prob)


class ScoreMatchingLoss:

    def __init__(self):
        pass

    @tf_var_scope
    def loss(self, log_prob, unstacked_data):
        """Compute the SM loss, looping over each dimension"""
        n, n_dims = tf.shape(unstacked_data[0])[0], len(unstacked_data)
        log_prob = tf.squeeze(log_prob)

        ssq = tf.constant(0., dtype=tf.float32)  # score function squared
        lap = tf.constant(0., dtype=tf.float32)  # laplacian
        for i in range(n_dims):
            feat = unstacked_data[i]
            score = tf.gradients(log_prob, feat)[0]
            ssq += tf.reduce_sum(score ** 2)
            lap += tf.reduce_sum(tf.gradients(tf.reduce_sum(score), feat)[0])

        return tf.cast(1 / n, tf.float32) * (lap + 0.5 * ssq)


# class MMDLoss:
#
#     def __init__(self, sigma):
#         self.sigma = sigma
#
#         params = {
#             "batch_size": 64,
#             "image_dim": 32 * 32 * 3,
#             "c": 3,
#             "h": 32,
#             "w": 32
#         }
#
#     def makeScaleMatrix(self, num_gen, num_orig):
#
#         # first 'N' entries have '1/N', next 'M' entries have '-1/M'
#         s1 = tf.constant(1.0 / num_gen, shape=[num_gen, 1])
#         s2 = -tf.constant(1.0 / num_orig, shape=[num_orig, 1])
#
#         return tf.concat([s1, s2], 0)
#
#     def computeMMD(self, x, gen_x, sigma = [1, 2, 4, 8, 16]):
#
#         x = slim.flatten(x)
#         gen_x = slim.flatten(gen_x)
#
#         # concatenation of the generated images and images from the dataset
#         # first 'N' rows are the generated ones, next 'M' are from the data
#         X = tf.concat([gen_x, x],0)
#
#         # dot product between all combinations of rows in 'X'
#         XX = tf.matmul(X, tf.transpose(X))
#
#         # dot product of rows with themselves
#         X2 = tf.reduce_sum(X * X, 1, keep_dims = True)
#
#         # exponent entries of the RBF kernel (without the sigma) for each
#         # combination of the rows in 'X'
#         # -0.5 * (x^Tx - 2 * x^Ty + y^Ty)
#         exponent = XX - 0.5 * X2 - 0.5 * tf.transpose(X2)
#
#         # scaling constants for each of the rows in 'X'
#         s = makeScaleMatrix(params['batch_size'], params['batch_size'])
#
#         # scaling factors of each of the kernel values, corresponding to the
#         # exponent values
#         S = tf.matmul(s, tf.transpose(s))
#
#         mmd_sq = 0
#
#         # for each bandwidth parameter, compute the MMD value and add them all
#         n = params['batch_size']
#         n_sq = float(n * n)
#         for i in range(len(sigma)):
#
#             # kernel values for each combination of the rows in 'X'
#             kernel_val = tf.exp(1.0 / sigma[i] * exponent)
#
#             mmd_sq += tf.reduce_sum(S * kernel_val)
#
#         return tf.sqrt(mmd_sq)
