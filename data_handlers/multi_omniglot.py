import numpy as np
import matplotlib.pyplot as plt
import os

from __init__ import project_root, density_data_root
from utils.plot_utils import disp_imdata, plot_hists_for_each_dim, save_fig
import data_handlers.data_utils as dutils
from data_handlers.base_dataset import Dataset
from scipy.stats import multivariate_normal


class MultiOmniglot:
    """
    The MultiOmniglot dataset -
    http://papers.nips.cc/paper/9692-wasserstein-dependency-measure-for-representation-learning.pdf
    """

    class Data:
        """
        Constructs the dataset.
        """

        def __init__(self, data, labels, stacked, n_imgs):
            self.x = data
            self.labels = labels
            self.stacked = stacked
            self.n_imgs = n_imgs

            self.ldj = 0
            self.N = self.x.shape[0]
            self.original_scale = 256.0

        def reverse_preprocessing(self, u):
            """Returns a single image, stacking characters as necessary depending on the form of the input"""
            if u.shape[-1] == 2:
                # spatialy stack x & y together
                x, y = np.split(u, 2, axis=-1)
                x, y = x[..., 0], y[..., 0]
                if self.stacked:
                    # x & y shapes: (n, 28, 28, n_imgs, 1)
                    x = MultiOmniglot.spatially_arrange_imgs(x, self.n_imgs, wmark_input=False)
                    y = MultiOmniglot.spatially_arrange_imgs(y, self.n_imgs, wmark_input=False)
                u = np.concatenate([x, y], axis=1)  # (n, 2 * n_imgs_sqrt * 28, n_imgs_sqrt * 28, 1)

            elif self.stacked:
                is_wmark_input = (len(u.shape) == 5)
                u = MultiOmniglot.spatially_arrange_imgs(u, self.n_imgs, wmark_input=is_wmark_input)

            return u

    def __init__(self, n_imgs, stacked=True, **kwargs):

        self.n_imgs = n_imgs
        self.n_sqrt = int(n_imgs**0.5)
        self.stacked = stacked
        trn, val, tst = self.load_datasets()

        self.trn = self.Data(*trn, stacked, n_imgs)
        self.val = self.Data(*val, stacked, n_imgs)
        self.tst = self.Data(*tst, stacked, n_imgs)

        self.image_size = self.trn.x.shape[1:]
        self.n_dims = np.prod(self.trn.x.shape[1:])

    def load_datasets(self):
        trn = self.load_set("trn")
        val = self.load_set("val")
        tst = self.load_set("tst")
        return trn, val, tst

    def load_set(self, which_set):
        f = np.load(density_data_root + 'omniglot/multiomniglot_{}_{}.npz'.format(which_set, self.n_imgs))
        xy, z = f["data"], f["labels"]

        xy /= 256.0

        if not self.stacked:  # arrange spatially
            assert xy.shape[1:] == (28, 28, self.n_imgs, 2), \
                "Event shape is {}, but expected (28, 28, {}, 2)".format(xy.shape[1:], self.n_imgs)

            x, y = np.split(xy, 2, -1)  # each (N, 28, 28, n_imgs, 1)

            x = self._spatially_arrange_imgs(x[..., 0])  # (N, 28*n_sqrt, 28*n_sqrt, 1)
            y = self._spatially_arrange_imgs(y[..., 0])  # (N, 28*n_sqrt, 28*n_sqrt, 1)

            xy = np.stack([x, y], axis=-1)  # (28*k, 28*k, 1, 2)

        return xy, z

    def _spatially_arrange_imgs(self, x):
        x = np.transpose(x, [0, 3, 1, 2])  # (N, n_sqrt**2, 28, 28)
        sub_imgs = np.split(x, self.n_sqrt, axis=1)
        sub_imgs = [np.reshape(i, (-1, 28*self.n_sqrt, 28)) for i in sub_imgs]
        x = np.concatenate(sub_imgs, axis=-1)  # (N, 28*n_sqrt, 28*n_sqrt)
        return np.expand_dims(x, -1)

    @staticmethod
    def spatially_arrange_imgs(x, n_imgs, wmark_input=True):
        # assuming inputs x are stacked along channels, rearrange them into a spatial grid
        n_sqrt = int(n_imgs**0.5)
        if wmark_input:
            transpose_idxs = [0, 1, 4, 2, 3]
            split_ax = 2
            batch_dims = x.shape[:2]
        else:
            transpose_idxs = [0, 3, 1, 2]
            split_ax = 1
            batch_dims = x.shape[:1]

        x = np.transpose(x, transpose_idxs)  # (*batch_dims, n_sqrt**2, 28, 28)
        sub_imgs = np.split(x, n_sqrt, axis=split_ax)  # n_sqrt tensors of shape (*batch_dims, n_sqrt, 28, 28)
        sub_imgs = [np.reshape(i, (*batch_dims, 28 * n_sqrt, 28)) for i in sub_imgs]
        x = np.concatenate(sub_imgs, axis=-1)  # (*batch_dims, 28*n_sqrt, 28*n_sqrt)
        return np.expand_dims(x, -1)

    def plot_datapoint(self, xy=None, i=0):
        fig_dir = project_root + "figs/omniglot/multiomniglot/"
        os.makedirs(fig_dir, exist_ok=True)

        if xy is None:
            xy = self.trn.x[i]
        x, y = np.split(xy, 2, axis=-1)
        x = x[..., 0]
        y = y[..., 0]

        if self.stacked:
            x = self._spatially_arrange_imgs(np.expand_dims(x, axis=0))[0]  # (28*k, 28*k, 1)
            y = self._spatially_arrange_imgs(np.expand_dims(y, axis=0))[0]  # (28*k, 28*k, 1)

        fig, axs = plt.subplots(1, 2)
        axs = axs.ravel()

        axs[0].imshow(np.squeeze(x), cmap="gray")
        axs[1].imshow(np.squeeze(y), cmap="gray")

        for ax in axs:
            ax.axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout()
        save_fig(fig_dir, "datapoint_{}".format(i), bbox_inches='tight')


def main():

    n_imgs = 9
    stacked = False
    data_obj = MultiOmniglot(n_imgs=n_imgs, stacked=stacked)

    data_obj.plot_datapoint(i=1)


if __name__ == "__main__":
    main()
