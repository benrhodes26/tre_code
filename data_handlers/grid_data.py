from __init__ import project_root, density_data_root
import data_handlers.data_utils as dutils
from itertools import product
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Affine2D
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from utils.misc_utils import *
from utils.plot_utils import *


class GridData:

    class Data:
        """
        Constructs the dataset.
        """

        def __init__(self, data, fit_cov_mat=False):
            self.x = data
            self.ldj = 0
            self.N = self.x.shape[0]
            if fit_cov_mat:
                self.cov_mat = np.cov(self.x, rowvar=False)
            self.original_scale = 1.0

    def __init__(self, grid_lim=None, **kwargs):
        trn, val = self.sample_data(), self.sample_data()

        self.nbins = 256
        grid_lim = [-0.999, 0.999] if grid_lim is None else grid_lim

        self.grid_lim_medium = np.array(grid_lim)
        self.grid_lim_small = (self.grid_lim_medium+0.05)/2
        self.grid_lim_large = 4 * self.grid_lim_medium

        self.tst_coords_medium, self.tst_grid_medium = self.make_tst_grid(self.grid_lim_medium)
        self.tst_coords_small, self.tst_grid_small = self.make_tst_grid(self.grid_lim_small)
        self.tst_coords_large, self.tst_grid_large = self.make_tst_grid(self.grid_lim_large)

        self.grid_shape = self.tst_grid_medium[0].shape
        try:
            self.grid_logp_medium = self.get_logdensity_on_tst_grid("medium", log=True)
            self.grid_logp_small = self.get_logdensity_on_tst_grid("small", log=True)
            self.grid_logp_large = self.get_logdensity_on_tst_grid("large", log=True)
        except NotImplementedError:
            pass

        self.trn = self.Data(trn, fit_cov_mat=True)
        self.val = self.Data(val)
        self.tst = self.Data(self.tst_coords_medium)

    def make_tst_grid(self, grid_lim):
        """

        :return:
            tst - array of shape (num_points_in_grid, 2)
            [x_grid_coords, y_grid_coords] - two arrays, each of shape (grid_width, grid_height)
        """
        low, high = grid_lim
        xi, yi = np.mgrid[low:high:self.nbins * 1j, low:high:self.nbins * 1j]
        tst = np.vstack([xi.flatten(), yi.flatten()]).T  # (grid_size, 2)
        return tst, [xi, yi]

    def get_logdensity_on_tst_grid(self, tst, log=True):
        raise NotImplementedError

    def sample_data(self):
        raise NotImplementedError

    def plot_sequences(self, data=None, dir_name=None, name="sequence_of_samples", s=0.05, label_type=None, gridsize="medium"):

        if data is None:
            data = self.trn.x.reshape(-1, 1, 2)
        if dir_name is None:
            dir_name = "{}figs/{}/".format(project_root, self.__repr__())

        n, k, d = data.shape
        max_n_cols = 4
        n_rows = int(np.ceil(k/max_n_cols))
        n_cols = k if n_rows == 1 else max_n_cols
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        axs = axs.ravel() if isinstance(axs, np.ndarray) else [axs]

        for i in range(k):
            axs[i].scatter(data[:, i, 0], data[:, i, 1], s=s)
            if label_type == "real_waymarks":
                label = r"$ \mathbf{x} \sim p_{%s}$" % i
            elif label_type == "sampled_waymarks":
                if i == 0:
                    label = r"$ \mathbf{x} \sim p_{%s}(.)$" % str(k-1)
                else:
                    label = r"$ \mathbf{x} \sim \tilde{p}_{%s}(. ; \mathbf{\theta})$" % str(k-i-1)

            if label_type:
                axs[i].set_title(label)
                axs[i].title.set_fontsize(20)

        for ax in axs:
            ax.set_xlim(*getattr(self, "grid_lim_{}".format(gridsize)))
            ax.set_ylim(*getattr(self, "grid_lim_{}".format(gridsize)))

        save_fig(dir_name, name, format="png" if os.path.isdir(local_pc_root) else "pdf")

    def plot_neg_energies(self, logp_experts, dir_name, fig_name, log_domain=False, gridsize="medium", **kwargs):
        """plot individual energies, as well as their sum, on the whole grid

        args
        energies: array of shape (tst_grid_size, n_experts)
        """
        grid = getattr(self, "tst_grid_{}".format(gridsize))

        try:
            n_experts = logp_experts.shape[-1]
            logp_experts = logp_experts.reshape(*self.grid_shape, n_experts)
        except:
            raise TypeError("`neg_energies' has first dimension of size {}, but should have same size as "
                            "self.grid_shape, which is: {}".format(logp_experts.shape[0], np.product(self.grid_shape)))

        plot_data_with_model = False
        n_figs = n_experts + 2 if plot_data_with_model else n_experts + 1
        axs, fig = create_subplot_with_max_num_cols(n_figs, max_n_cols=4)
        ax_num = 0

        true_logp = getattr(self, "grid_logp_{}".format(gridsize))
        true_logp = true_logp if log_domain else np.exp(true_logp)
        title = r"$p_{data}(\mathbf{x}) \equiv p_{0}(\mathbf{x})$" if not log_domain else \
            r"$\log p_{data}(\mathbf{x}) \equiv \log p_{0}(\mathbf{x})$"

        if plot_data_with_model:
            self.make_custom_imshow_plot(axs[ax_num], fig, grid, true_logp, title)
            self.place_textbox(axs[ax_num], r"$\approx$")
            ax_num += 1
        else:
            data_dir = dir_name + "data_density/"
            dfig_name = "data_log_density_{}".format(gridsize) if log_domain else "data_density_{}".format(gridsize)
            os.makedirs(data_dir, exist_ok=True)
            if not os.path.isfile(path_join(data_dir, dfig_name + ".pdf")):
                fig2, ax = plt.subplots(1, 1)
                self.make_custom_imshow_plot(ax, fig2, grid, true_logp, title)
                self.place_textbox(ax, r"$\approx$")
                save_fig(data_dir, dfig_name, fig=fig2)

        self.plot_product_of_experts(fig, axs[ax_num], log_domain, logp_experts, grid)
        ax_num += 1

        for i in range(ax_num, n_experts+ax_num):
            j = i - 2 if plot_data_with_model else i - 1
            self.plot_expert(fig, axs[i], i, j, n_experts + ax_num - 1, log_domain, logp_experts, grid)

        for ax in axs:
            ax.set_xlim(*getattr(self, "grid_lim_{}".format(gridsize)))
            ax.set_ylim(*getattr(self, "grid_lim_{}".format(gridsize)))
            ax.title.set_fontsize(20)

        fig.tight_layout()
        # save_fig(dir_name, fig_name, format="png" if os.path.isdir(local_pc_root) else "pdf")
        save_fig(dir_name, fig_name, fig=fig)

    def plot_product_of_experts(self, fig, ax, log, logp_experts, grid):

        title = r"$ \log \tilde{p}_{model}(\mathbf{x}; \mathbf{\theta})$" if log else\
            r"$ \tilde{p}_{model}(\mathbf{x}; \mathbf{\theta})$"

        total_energy = np.sum(logp_experts, axis=-1)
        e = total_energy if log else np.exp(total_energy)
        self.make_custom_imshow_plot(ax, fig, grid, e, title)
        self.place_textbox(ax, r"$=$")

    def plot_expert(self, fig, ax, i, j, noise_idx, log_domain, logp_experts, grid):
        if log_domain:
            title = r"$\log p_{%s}(\mathbf{x})$" % j if i == noise_idx\
                else r"$\log r_{%s}(\mathbf{x}; \mathbf{\theta}_{%s})$" % (j, j)
        else:
            title = r"$p_{%s}(\mathbf{x})$" % j if i == noise_idx \
                else r"$r_{%s}(\mathbf{x}; \mathbf{\theta}_{%s})$" % (j, j)

        e = logp_experts[:, :, j] if log_domain else np.exp(logp_experts[:, :, j])

        e[e > 10**32] = 10**32
        self.make_custom_imshow_plot(ax, fig, grid, e, title)

        if i < noise_idx:
            self.place_textbox(ax, r"$\plus$ " if log_domain else r"$\times$")

    def plot_true_density(self, log=False, gridsize="medium"):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        name = "energy" if log else "density"
        fig.suptitle("ground truth data {}".format(name))

        vals = getattr(self, "grid_logp_{}".format(gridsize))
        vals = vals if log else np.exp(vals)
        levels = MaxNLocator(nbins=20).tick_values(-20.0, vals.max())
        cmap = plt.get_cmap('viridis')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        im = ax.pcolormesh(self.tst_grid[0], self.tst_grid[1], vals, norm=norm)
        ax.set_title("ground truth")
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=14)

        ax.set_xlim(*getattr(self, "grid_lim_{}".format(gridsize)))
        ax.set_ylim(*getattr(self, "grid_lim_{}".format(gridsize)))

        dir_name = "{}figs/{}/".format(project_root, self.__repr__())
        save_fig(dir_name, "true_data_{}".format(name))

    def make_custom_imshow_plot(self, ax, fig, grid, x, title):
        norm = self.get_cmap_norm(x)

        im = ax.imshow(x.T, interpolation='nearest', norm=norm, origin="lower",
                       extent=[grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()])
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=14)

    @staticmethod
    def get_cmap_norm(x):
        levels = MaxNLocator(nbins=50).tick_values(x.min(), x.max())
        cmap = plt.get_cmap('viridis')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        return norm

    @staticmethod
    def place_textbox(ax, str):
        """place a text box containing a string to top right of figure"""
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(1.35, 1.12, str, transform=ax.transAxes, fontsize=20, verticalalignment='top', bbox=props)

    def __repr__(self):
        return "2d_GridData"


class MoG(GridData):
    """
    2d grid Mixture of Gaussians
    """

    def __init__(self, n_gaussians_per_dim, samples_per_gaussian, covar_mult, **kwargs):
        self.n_gaussians_per_dim = n_gaussians_per_dim
        self.outer_gauss_coords = [0.2, 0.8]
        self.n_samples_per_gaussian = samples_per_gaussian
        self.covar_mult = covar_mult
        super().__init__()

    def eval_density(self, x):
        return np.mean(np.array([g.pdf(x) for g in self.gaussians]), axis=0)

    def get_logdensity_on_tst_grid(self, gridsize, log=True):
        grid = getattr(self, "tst_grid_{}".format(gridsize))
        tst = getattr(self, "tst_coords_{}".format(gridsize))

        mog_grid_vals = logsumexp(np.array([g.logpdf(tst) for g in self.gaussians]), axis=0)
        mog_grid_vals -= np.log(len(self.gaussians))
        mog_grid_vals = mog_grid_vals if log else np.exp(mog_grid_vals)
        mog_grid_vals = mog_grid_vals.reshape(grid[0].shape)

        return mog_grid_vals

    def sample_data(self):
        if self.n_gaussians_per_dim == 1:
            centres = [(0, 0)]
            var = 0.2
        else:
            x_coord = np.linspace(*self.outer_gauss_coords, self.n_gaussians_per_dim)
            centres = product(x_coord, x_coord)

            dist = x_coord[1] - x_coord[0]
            var = (dist / 7)**2  # ensures that the gaussians are well-separated

        params = [(np.array(c), np.array([[var, self.covar_mult*var], [self.covar_mult*var, var]])) for c in centres]
        self.gaussians = [multivariate_normal(*p) for p in params]
        samples = np.array([[g.rvs() for _ in range(self.n_samples_per_gaussian)] for g in self.gaussians])

        return samples.reshape(-1, 2)

    def __repr__(self):
        return "2d_mog"


class Spiral(GridData):
    """
    2d spiral data set, constructed by adding gaussian noise to spiral curve
    """

    def __init__(self, grid_lim=None, mode="broad", num_samples=100000, add_broad_noise=False, **kwargs):
        """

        :param mode: either 'broad', 'peaked', 'degenerate' or 'mixture'
            determines the peakiness of the gaussians densities that form the spiral.
            'degenerate' means extremely peaked, such that it approximates a 1d curve.
            'mixture' means that the centre-most gaussians are peaked, but as we move
            outwards along the spiral, they become broader.
        """
        self.mode = mode
        self.total_num_samples = num_samples
        self.add_broad_noise = add_broad_noise
        self.init_gaussians(mode)
        super().__init__(grid_lim)

    def init_gaussians(self, mode):

        if mode == "broad":
            self.num_gaussians = 100
            vars = [1e-3]*self.num_gaussians

        elif mode == "peaked":
            self.num_gaussians = 1000
            vars = [1e-5]*self.num_gaussians

        elif mode == "degenerate":
            self.num_gaussians = 1000
            vars = [1e-10]*self.num_gaussians

        elif mode == "mixture":
            self.num_gaussians = 500
            vars = list(np.linspace(0.00005, 0.0005, num=self.num_gaussians))

        else:
            raise ValueError("Unknown mode: {}. Choose either 'broad', 'peaked', or 'mixture'".format(mode))

        centres = self.get_gauss_centres()
        # centres *= 0.9  # concentrate the spiral
        self.gaussians = [multivariate_normal(mean=xy, cov=var * np.identity(2)) for var, xy in zip(vars, centres)]
        if self.add_broad_noise:
            self.gaussians.append(multivariate_normal(mean=np.ones(2), cov=0.3*np.array([[1, 0.0], [0.0, 1]])))
        self.num_samples_per_gauss = self.total_num_samples // self.num_gaussians
        print("Sample neg entropy: {}".format(-self.empirical_entropy()))

    def get_gauss_centres(self):
        thetas = np.linspace(0, (6 * np.pi) ** 2, self.num_gaussians)
        thetas = np.sqrt(thetas)
        x_cors = 0.0 + (1 / 40) * thetas * np.cos(thetas)
        y_cors = 0.0 + (1 / 40) * thetas * np.sin(thetas)
        centres = np.vstack((x_cors, y_cors)).T
        return centres

    def eval_density(self, x, log=True):
        val = logsumexp(np.array([g.logpdf(x) for g in self.gaussians]), axis=0)
        val -= np.log(len(self.gaussians))
        val = val if log else np.exp(val)
        return val

    def get_logdensity_on_tst_grid(self, gridsize, log=True):
        grid = getattr(self, "tst_grid_{}".format(gridsize))
        tst = getattr(self, "tst_coords_{}".format(gridsize))

        mog_grid_vals = logsumexp(np.array([g.logpdf(tst) for g in self.gaussians]), axis=0)
        mog_grid_vals -= np.log(len(self.gaussians))

        mog_grid_vals = mog_grid_vals if log else np.exp(mog_grid_vals)
        mog_grid_vals = mog_grid_vals.reshape(grid[0].shape)
        return mog_grid_vals

    def sample_data(self):
        samples = np.array([[g.rvs() for _ in range(self.num_samples_per_gauss)] for g in self.gaussians])
        return samples.reshape(-1, 2)

    def empirical_entropy(self):
        # should really calculate this analytically...
        samples = self.sample_data()
        logp = self.eval_density(samples, log=True)
        return -np.mean(logp)

    def __repr__(self):
        return "2d_spiral"


if __name__ == "__main__":
    spiral = Spiral(mode="degenerate")
    spiral.plot_sequences()
    spiral.plot_true_density()
    spiral.plot_true_density(log=True)
    print("Entropy is: {}".format(spiral.empirical_entropy()))
