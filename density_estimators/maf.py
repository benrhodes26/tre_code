import tensorflow_probability as tfp

from density_estimators.mades import MogMade, residual_mog_made_template, residual_made_template
from density_estimators.gauss_copula import GaussianCopulaFromSplines
from utils.tf_utils import *

tfb = tfp.bijectors
tfd = tfp.distributions


# class CouplingFlow(tf.keras.Model):
#
#     def __init__(self):
#         super(CouplingFlow, self).__init__()
#         self.block_1 = ResNetBlock()
#         self.block_2 = ResNetBlock()
#         self.global_pool = layers.GlobalAveragePooling2D()
#         self.classifier = Dense(num_classes)
#
#     def call(self, inputs):
#         x = self.block_1(in
# puts)
#         x = self.block_2(x)
#         x = self.global_pool(x)
#         return self.classifier(x)
#
#
# class CouplingLayer(layers.Layer):
#
#     def __init__(self, type="rational_quadratic", units=32, rqs_nbins=None, **kwargs):
#         super(CouplingLayer, self).__init__(**kwargs)
#         self.type = type
#         self.units = units
#         self.rqs_n_bins = rqs_nbins
#
#     def build(self, input_shape):
#         if self.type == "rational_quadratic":
#             self.bijector_fn = SplineBijectorFn(input_shape, self.units, nbins=self.rqs_n_bins)
#         else:
#             raise NotImplementedError
#
#     def call(self, inputs):
#         return tfb.RealNVP(

