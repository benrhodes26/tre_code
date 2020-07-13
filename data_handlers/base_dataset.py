import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from scipy.linalg import svd as scipy_svd
from sklearn.decomposition import PCA

import data_handlers.data_utils as dutils
from __init__ import project_root, density_data_root
from scipy.stats import multivariate_normal
from utils.plot_utils import disp_imdata
import utils.misc_utils as misc_utils


class Dataset:

    def __init__(self,
                 x,
                 dequantize=False,
                 original_scale=1.0,
                 logit=False,
                 logit_alpha=1e-6,
                 pca=False,
                 reconstructed_pca=False,
                 zca=False,
                 standardize=False,
                 img_shape=None,
                 n_pca_components = None,
                 flip_augment = False,
                 shift = None,
                 pca_obj = None,
                 zca_params = None,
                 scale = None,
                 labels = None,
                 fit_cov_mat = False,
                 pca_orig_img_shape=None):

        rng = np.random.RandomState(1234)

        if dequantize: x = self._dequantize(x, rng, original_scale)
        if flip_augment: x = self._flip_augmentation(x)
        if flip_augment: labels = np.hstack([labels, labels])

        self.img_shape = img_shape if not pca else pca_orig_img_shape
        self.event_shape = self.img_shape if self.img_shape is not None else [np.prod(x.shape[1:])]
        x = x.reshape((x.shape[0], *self.event_shape))

        self.logit_alpha = logit_alpha
        self.original_scale = original_scale

        self.N = x.shape[0]  # number of datapoints
        self.n_dims = np.prod(self.event_shape)

        if reconstructed_pca and fit_cov_mat:
            # fit the cov matrix to the original data
            x_flat = x.reshape(self.N, -1)
            x_flat = self._logit_transform(x_flat) if logit else x_flat
            x_flat = x_flat - x.mean(axis=0) if shift else x_flat
            self.cov_mat = np.cov(x_flat, rowvar=False)

        x = self.forward_preprocessing(x,
                                       logit,
                                       shift,
                                       pca,
                                       reconstructed_pca,
                                       n_pca_components,
                                       pca_obj,
                                       zca,
                                       zca_params,
                                       standardize,
                                       scale
                                       )

        if not reconstructed_pca and fit_cov_mat:
            x_flat = x.reshape(self.N, -1)
            self.cov_mat = np.cov(x_flat, rowvar=False)
            # mvn = multivariate_normal(np.mean(x_flat, axis=0), self.cov_mat)
            # mvn_loglik = np.mean(mvn.logpdf(x_flat))
            # print("MVN loglik: ", mvn_loglik)
            # mvn_bpd = misc_utils.convert_to_bits_per_dim(mvn_loglik + np.mean(self.ldj), self.n_dims, self.original_scale)
            # print("MVN bpd: ", mvn_bpd)

        self.x = x
        self.labels = labels

    def forward_preprocessing(self,
                              x,
                              logit,
                              shift,
                              pca,
                              reconstructed_pca,
                              n_pca_components,
                              pca_obj,
                              zca,
                              zca_params,
                              standardize,
                              scale
                              ):

        self.ldj = np.zeros(len(x))  # default value of logdetjacobian (useful for density-estimation)

        self.logit = logit
        if logit:
            s = lambda x: self._scale_x_with_alpha(x.reshape(len(x), -1))
            get_logit_ldj = lambda x: np.sum(np.log(1/s(x) + 1/(1-s(x))) + np.log(1 - 2 * self.logit_alpha), axis=-1)
            logit_ldj = get_logit_ldj(x)
            x = self._logit_transform(x)

            # equivalently:
            # x = self._logit_transform(x)
            # sig = lambda x: misc_utils.sigmoid(x)
            # get_logit_ldj = lambda x: np.sum(-np.log(sig(x)) - np.log(1 - sig(x)) + np.log(1 - 2 * self.logit_alpha), axis=-1)
            # logit_ldj = get_logit_ldj(x.reshape(len(x), -1))

            print("logit ldj:", np.mean(logit_ldj))
            self.ldj += logit_ldj
        else:
            self.logit_shift = self._logit_transform(x).mean(axis=0)

        # zero-centre each feature
        self.shift = x.mean(axis=0) if shift is None else shift
        x -= self.shift

        if pca or reconstructed_pca:
            print("Appling PCA to data")
            x, self.pca_obj = self.apply_pca(x, n_pca_components, pca_obj)
            if reconstructed_pca:
                x = self.pca_obj.inverse_transform(x)
                x = x.reshape(x.shape[0], *self.event_shape)
        else:
            self.pca_obj = None

        if zca:
            if pca or reconstructed_pca: raise ValueError("Do not apply zca after pca")
            print("Appling ZCA to data")

            if zca_params is None:
                x, zca_rot_mat, zca_scale, zca_ldj = self.apply_zca(x)
            else:
                x, zca_rot_mat, zca_scale, zca_ldj = self.apply_zca(x, *zca_params)
            self.zca_params = [zca_rot_mat, zca_scale, zca_ldj]
            self.ldj += zca_ldj
            print("zca ldj: {}".format(zca_ldj.mean()))
        else:
            self.zca_params = None

        if standardize:
            if zca: raise ValueError("No use in applying standardization after zca")
            print("Standardizing each feature to unit-variance")

            self.scale = x.std(axis=0) if scale is None else scale
            x, scale_ldj = self.scale_data(x, self.scale)
            self.ldj += scale_ldj
            print("scale ldj:", np.mean(scale_ldj))
        else:
            self.scale = None

        print("Average logdetjac of preprocessing is: ", self.ldj.mean())
        return x

    def reverse_preprocessing(self, x):
        y = deepcopy(x)
        if self.scale is not None:
            print("Rescaling data as part of pre-processing")
            y *= self.scale

        if self.zca_params is not None:
            print("undoing zca transform as part of pre-processing")
            y = self.reverse_zca(y, *self.zca_params)

        if self.pca_obj is not None and x.shape[1:] != tuple(self.event_shape):
            print("Reconstructing PCA data as part of pre-processing")
            y = self.pca_obj.inverse_transform(y)
            y = y.reshape(-1, *self.img_shape)

        y += self.shift

        if self.logit:
            y = self.logit_inv(y)

        return y

    @staticmethod
    def _dequantize(x, rng, scale):
        """
        Adds uniform noise to pixels to dequantize them.
        """
        return x + rng.rand(*x.shape) / scale

    def _logit_transform(self, x):
        """
        Transforms pixel values with logit to be unconstrained.
        """
        return misc_utils.logit(self._scale_x_with_alpha(x))


    def _scale_x_with_alpha(self, x):
        return self.logit_alpha + (1 - 2 * self.logit_alpha) * x

    def logit_inv(self, x):
        """
        inverts the logit transform.
        """
        return (misc_utils.sigmoid(x) - self.logit_alpha) / (1 - 2 * self.logit_alpha)

    @staticmethod
    def apply_pca(x, n_pca_components, pca_obj):
        x = x.reshape(x.shape[0], -1)
        if pca_obj is None:
            print("Fitting PCA transform to train data!")
            pca_obj = PCA(n_components=n_pca_components, svd_solver='full')
            pca_obj.fit(x)

        x = pca_obj.transform(x)
        return x, pca_obj

    @staticmethod
    def apply_zca(x, U = None, S = None, ldj = None):
        """Assumes x has been zero-centered"""
        x_shp = x.shape
        x = x.reshape(x_shp[0], -1)

        # singular value decomposition
        if U is None:
            assert S is None, "ZCA rotation matrix is None, but scale vector is not None"
            cov = np.cov(x, rowvar=False)  # cov is (N, N)
            cov = cov.astype(np.float32)
            # U, S, _ = np.linalg.svd(cov)  # U is (N, N), S is (N,)
            U, S, _ = scipy_svd(cov, overwrite_a=True)  # U is (N, N), S is (N,)

        # build the ZCA matrix
        epsilon = 1e-5
        zca_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))  # (N,N)

        # transform the image data
        z = np.dot(x, zca_matrix)  # zca is (N, d)

        if ldj is None:
            sgn, ldj = np.linalg.slogdet(zca_matrix)
            assert sgn == 1, "Sign of logdetjacobian of zca matrix is not positive"

        return z.reshape(x_shp), U, S, ldj

    @staticmethod
    def reverse_zca(z, U, S, _):
        z_shp = z.shape
        z = z.reshape(z_shp[0], -1)

        epsilon = 1e-5
        inv_zca_matrix = np.dot(U, np.dot(np.diag(np.sqrt(S + epsilon)), U.T))

        x = np.dot(z, inv_zca_matrix)

        return x.reshape(z_shp)

    @staticmethod
    def standardize_data(x, shift, scale):
        if not isinstance(shift, np.ndarray):
            shift = np.ones_like(x[0]) * shift
            scale = np.ones_like(x[0]) * scale

        x = (x - shift) / scale
        ldj_per_x = np.zeros(len(x))
        ldj_per_x -= np.sum(np.log(scale))  # log det jacobian of standardisation
        return x, ldj_per_x

    @staticmethod
    def scale_data(x, scale):
        if not isinstance(scale, np.ndarray):
            scale = np.ones_like(x[0]) * scale

        x /= scale
        ldj = np.zeros(len(x))
        ldj -= np.sum(np.log(scale))  # log det jacobian of standardisation
        return x, ldj

    @staticmethod
    def _flip_augmentation(x):
        """
        Augments dataset x with horizontal flips.
        """
        x_flip = x[:, :, ::-1, :]
        return np.vstack([x, x_flip])
