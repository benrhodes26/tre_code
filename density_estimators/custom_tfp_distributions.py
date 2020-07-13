import tensorflow_probability as tfp

from utils.tf_utils import *

tfb = tfp.bijectors
tfd = tfp.distributions

class logit(tfp.bijectors.Bijector):

    def __init__(self, inverse_min_event_ndims, alpha, name="logit"):
        self.alpha = alpha
        super(logit, self).__init__(inverse_min_event_ndims=inverse_min_event_ndims, name=name)

    def _forward(self, x):
        x = self.scale_x_with_alpha(x)
        return tf.log(x / (1.0 - x))

    def _inverse(self, x):
        x = 1 / (1 + tf.exp(-x))
        return (x - self.alpha) / (1 - 2 * self.alpha)

    def _forward_log_det_jacobian(self, x):
        x = tf.reshape(x, [shape_list(x)[0], -1])
        s_x = self.scale_x_with_alpha(x)
        ldj = tf.reduce_sum(tf.log(1/s_x + 1/(1-s_x)) + tf.log(1 - 2*self.alpha), axis=-1)
        return ldj

    def scale_x_with_alpha(self, x):
        return self.alpha + (1 - 2 * self.alpha) * x
