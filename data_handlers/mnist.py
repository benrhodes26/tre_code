import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt

from __init__ import project_root, density_data_root
from utils.plot_utils import disp_imdata, plot_hists_for_each_dim
import data_handlers.data_utils as dutils
from data_handlers.base_dataset import Dataset
from scipy.stats import multivariate_normal


class MNIST:
    """
    The MNIST dataset of handwritten digits.
    """

    def __init__(self, crop_n=None, class_label=None, n_pca_components=100, percent_excluded=None, flow_type=None, **kwargs):

        logit_alpha = 1.0e-6
        original_scale = 256.0

        trn, val, tst = self.load_datasets(class_label, percent_excluded, flow_type)
        if crop_n:
            trn = [dutils.crop_img(trn[0], (28, 28, 1), crop_n), trn[1]]
            val = [dutils.crop_img(val[0], (28, 28, 1), crop_n), val[1]]
            tst = [dutils.crop_img(tst[0], (28, 28, 1), crop_n), tst[1]]

        self.trn = Dataset(x=trn[0],
                           original_scale=original_scale,
                           logit_alpha=logit_alpha,
                           n_pca_components=n_pca_components,
                           labels=trn[1],
                           fit_cov_mat=True,
                           **kwargs)

        self.val = Dataset(x=val[0],
                           original_scale=original_scale,
                           logit_alpha=logit_alpha,
                           n_pca_components=n_pca_components,
                           shift=self.trn.shift,
                           pca_obj=self.trn.pca_obj,
                           zca_params=self.trn.zca_params,
                           scale=self.trn.scale,
                           labels=val[1],
                           **kwargs)

        self.tst = Dataset(x=tst[0],
                           original_scale=original_scale,
                           logit_alpha=logit_alpha,
                           n_pca_components=n_pca_components,
                           shift=self.trn.shift,
                           pca_obj=self.trn.pca_obj,
                           zca_params=self.trn.zca_params,
                           scale=self.trn.scale,
                           labels=tst[1],
                           **kwargs)

        self.n_dims = np.prod(self.trn.x.shape[1:])
        self.image_size = [int(np.sqrt(self.n_dims))] * 2

    @staticmethod
    def load_datasets(class_label, percent_excluded=None, flow_type=None):

        f = gzip.open(density_data_root + 'mnist/mnist.pkl.gz', 'rb')
        trn, val, tst = pickle.load(f, encoding='latin1')
        f.close()

        if percent_excluded is not None:
            trn, val, tst = dutils.load_percent_excluded_data(trn, val, tst, "mnist", percent_excluded, flow_type)

        if class_label is not None:
            get_class = lambda x: [x[0][x[1] == class_label], x[1][x[1] == class_label]]
            trn, val, tst = get_class(trn), get_class(val), get_class(tst)

        return trn, val, tst


def main():
    dataset_name = "mnist"
    dir_name = "figs/{}/img_data/".format(dataset_name)
    # dataset_name = "pca_cropped_mnist"
    img_shape = [28, 28, 1]
    pca, n_components = False, 2
    reconstructed_pca = False
    class_label = None
    flip_augment = False

    mnist_obj = MNIST(dequantize=True, logit=True, class_label=class_label, flip_augment=flip_augment,
                      n_pca_components=n_components, pca=pca, reconstructed_pca=reconstructed_pca, img_shape=img_shape)

    # for i in [[0,0], [10, 10], [14, 14], [14, 20], [16, 8], [19, 14]]:
    #     dutils.show_pixel_histograms(mnist_obj.trn.x, i)

    def print_mean_and_std(dset, mode= "train"):
        # print("{} mean ".format(mode), dset.x.mean(axis=0))
        # print("{} std: ".format(mode), dset.x.std(axis=0))
        n_dims = np.prod(dset.x.shape[1:])
        print("{} av log lik under standard normal:".format(mode),
              np.mean(multivariate_normal.logpdf(dset.x.reshape(dset.x.shape[0], -1), mean=np.zeros(n_dims), cov=np.diag(np.ones(n_dims))))
              )
        print("{} norm: ".format(mode), np.mean(np.sum(dset.x**2, axis=tuple(range(1, len(dset.x.shape))))**0.5))

    def show_imgs(mnist_obj, dir_name, name, encode=True, decode=True, disp_gauss_noise=False):
        if encode and not reconstructed_pca:
            if pca:
                x = mnist_obj.trn.x
                plot_hists_for_each_dim(n_dims_to_plot=x.shape[1],
                                        data=x,
                                        dir_name=project_root + dir_name,
                                        filename="pca_hists",
                                        include_scatter=True
                                        )
            else:
                std = mnist_obj.trn.x.std()
                vmin, vmax = [-3 * std, 3 * std]
                # vmin, vmax = [0, 1]
                disp_imdata(mnist_obj.trn.x, dataset_name, num_pages=1, dir_name=project_root + dir_name, name=name, vminmax=[vmin, vmax])
                if disp_gauss_noise:
                    gauss_noise = np.random.normal(0, 1, size=(50, *img_shape))
                    disp_imdata(gauss_noise, dataset_name, num_pages=1, dir_name=project_root + dir_name, name=name + "_gauss_noise", vminmax=[vmin, vmax])
        if decode:
            recon_trn = mnist_obj.trn.reverse_preprocessing(mnist_obj.trn.x)
            # std = recon_trn.std()
            # vmin, vmax = [-3 * std, 3 * std]
            vmin, vmax = [0, 1]
            disp_imdata(recon_trn, dataset_name, num_pages=1, dir_name=project_root + dir_name, name="reverse " + name, vminmax=[vmin, vmax])
            if disp_gauss_noise:
                gauss_noise = np.random.normal(0, 1, size=(50, *img_shape))
                recon_gauss_noise = mnist_obj.trn.reverse_preprocessing(gauss_noise)
                disp_imdata(recon_gauss_noise, dataset_name, num_pages=1, dir_name=project_root + dir_name,
                            name="reverse " + name + "_gauss_noise", vminmax=[vmin, vmax])


    gauss_samples = multivariate_normal(np.zeros(mnist_obj.n_dims), mnist_obj.trn.cov_mat).rvs(49)
    disp_imdata(gauss_samples.reshape(-1, 28, 28, 1), dataset_name, num_pages=1,
                dir_name=project_root + dir_name, name="mvn_samples", vminmax=[0, 1])

    # print_mean_and_std(mnist_obj.trn, "trn")
    # print_mean_and_std(mnist_obj.val, "val")
    # print_mean_and_std(mnist_obj.tst, "tst")

    show_imgs(mnist_obj, dir_name, "imgs")


if __name__ == "__main__":
    main()
