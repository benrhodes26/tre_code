import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import block_diag
from scipy.optimize import fsolve
from scipy.stats import multivariate_normal
from utils.misc_utils import kl_between_two_gaussians


class GAUSSIANS:
    """
    Synthetic lines dataset
    """

    class Data:
        """
        Constructs the dataset.
        """

        def __init__(self, data, vars, cov_mat):
            self.x = data
            self.ldj = 0
            self.N = self.x.shape[0]     # number of datapoints

            self.variances = vars
            self.cov_matrix = cov_mat
            self.original_scale = 1.0

    def __init__(self, n_samples, n_dims=80, true_mutual_info=None, mean=None, std=None, base_mi=None, **kwargs):

        if (mean is not None) or (std is not None) or (base_mi is not None):
            assert (mean is not None) and (std is not None) and (base_mi is not None)
            assert true_mutual_info is None, "Can't specify mean/std AND true_mutual_info"
        else:
            assert true_mutual_info is not None, "Must specify MI if mean+std are unspecified"

        self.n_dims = n_dims
        self.means = np.ones(n_dims) * mean
        self.variances = np.ones(n_dims) * std**2

        if true_mutual_info is not None:
            self.rho = self.get_rho_from_mi(true_mutual_info, n_dims)  # correlation coefficient
            self.cov_matrix = block_diag(*[[[1, self.rho], [self.rho, 1]] for _ in range(n_dims // 2)])
        else:
            self.rho = self.get_rho_from_mi(base_mi, n_dims)  # correlation coefficient
            self.cov_matrix = block_diag(*[[[1, self.rho], [self.rho, 1]] for _ in range(n_dims // 2)])

        self.denom_cov_matrix = np.diag(self.variances)

        trn, val, tst = self.sample_data(n_samples), self.sample_data(n_samples), self.sample_data(n_samples)

        self.trn = self.Data(trn, self.variances, self.cov_matrix)
        self.val = self.Data(val, self.variances, self.cov_matrix)
        self.tst = self.Data(tst, self.variances, self.cov_matrix)

        self.n_dims = trn.shape[1]

    @staticmethod
    def get_rho_from_mi(mi, n_dims):
        """Get correlation coefficient from true mutual information"""
        x = (4 * mi) / n_dims
        return (1 - np.exp(-x)) ** 0.5  # correlation coefficient

    def sample_data(self, n_samples):
        return self.sample_gaussian(n_samples, self.cov_matrix)

    def sample_denominator(self, n_samples):
        return self.sample_gaussian(n_samples, self.denom_cov_matrix)

    def sample_gaussian(self, n_samples, cov_matrix):
        prod_of_marginals = multivariate_normal(mean=self.means, cov=cov_matrix)
        return prod_of_marginals.rvs(n_samples)

    def numerator_log_prob(self, u):
        mvn = multivariate_normal(mean=self.means, cov=self.cov_matrix)
        log_probs = mvn.logpdf(u)
        return log_probs

    def denominator_log_prob(self, u):
        prod_of_marginals = multivariate_normal(mean=np.zeros(self.n_dims), cov=self.denom_cov_matrix)
        return prod_of_marginals.logpdf(u)

    def empirical_mutual_info(self, samples=None):
        if samples is None:
            samples = self.sample_data(100000)
        return np.mean(self.numerator_log_prob(samples) - self.denominator_log_prob(samples))

    def determine_waymark_spacing(self, target_kl, logger=None):

        target_kl /= (self.n_dims / 2)  # target kl for correlated bivariate distributions
        covs = [self.rho]
        while True:

            def kl_as_fn_of_cov(sqrt_cov):
                numer_cov = self.make_2d_cov_matrix(1.0, covs[-1])
                denom_cov = self.make_2d_cov_matrix(1.0, sqrt_cov**2)
                kl = kl_between_two_gaussians(numer_cov, denom_cov)
                residual = kl - target_kl
                return residual

            def kl_between_waymark_and_base(cov):
                numer_cov = self.make_2d_cov_matrix(1.0, cov)
                denom_cov = self.make_2d_cov_matrix(1.0, 0.0)
                return kl_between_two_gaussians(numer_cov, denom_cov)

            estimated_cov = np.inf
            j = 1
            finished = False
            while estimated_cov >= covs[-1]:
                if kl_between_waymark_and_base(covs[-1]) <= target_kl:
                    finished = True
                    break

                estimated_sqrt_cov = fsolve(kl_as_fn_of_cov, (covs[-1] - 0.0001 * j)**0.5)[0]
                estimated_cov = estimated_sqrt_cov**2
                j += 1
                if j >= 100:
                    raise ValueError("Unable to solve for covariance of a waymark")

            if finished:
                covs.append(0)
                break
            else:
                covs.append(estimated_cov)

        if logger:
            logger.info("estimated sequence of covs: {}".format(covs))
            logger.info("estimated diffs in sequence of covs: {}".format([cov1 - cov2 for cov1, cov2 in zip(covs[:-1], covs[1:])]))

        consecutive_kls = []
        kls_wrt_numer_dist = []
        for i in range(len(covs)-1):
            w1_cov = self.make_2d_cov_matrix(1.0, covs[i])
            w2_cov = self.make_2d_cov_matrix(1.0, covs[i+1])
            consecutive_kls.append(kl_between_two_gaussians(w1_cov, w2_cov))

            numer_cov = self.make_2d_cov_matrix(1.0, self.rho)
            kls_wrt_numer_dist.append(kl_between_two_gaussians(numer_cov, w1_cov))

        kls_wrt_numer_dist.append(kl_between_two_gaussians(numer_cov, w2_cov))

        if logger:
            logger.info("KLs w.r.t numerator dist: {}".format(kls_wrt_numer_dist))
            logger.info("consecutive waymark KLs: {}".format(consecutive_kls))

        return covs

    @staticmethod
    def make_2d_cov_matrix(var, cov):
        if not isinstance(var, np.ndarray): var = np.array([var, var])
        return np.array([[var[0], cov], [cov, var[1]]], dtype=np.float32)

    def show_pixel_histograms(self, split, pixel=None):
        """
        Shows the histogram of pixel values, or of a specific pixel if given.
        """

        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        if pixel is None:
            data = data_split.x.flatten()
        else:
            row, col = pixel
            idx = row * self.image_size[0] + col
            data = data_split.x[:, idx]

        n_bins = int(np.sqrt(data_split.N))
        fig, ax = plt.subplots(1, 1)
        ax.hist(data, n_bins, density=True)
        plt.show()


def main():
    n, d = 100000, 80
    true_mi = 40

    dataset = GAUSSIANS(n_samples=n, n_dims=d, true_mutual_info=true_mi)
    print("True MI is {}, empirical MI is: {}".format(dataset.true_mutual_info, dataset.empirical_mutual_info()))

    return dataset


if __name__ == "__main__":
    main()
