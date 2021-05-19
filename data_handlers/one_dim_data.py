from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
from utils.misc_utils import *
from utils.plot_utils import *


class OneDimData:

    class Data:
        """
        Constructs the dataset.
        """

        def __init__(self, data):
            self.x = data
            self.ldj = 0
            self.original_scale = 1.0
            self.N = self.x.shape[0]

    def __init__(self, scale=1.0):
        self.dims = 1
        trn, val = self.sample_data(), self.sample_data()

        grid_lim = [-0.999, 0.999]

        self.grid_lim_medium = np.array(grid_lim)
        self.grid_lim_small = self.grid_lim_medium*scale
        self.grid_lim_large = 4 * self.grid_lim_medium

        self.tst_grid_medium = np.linspace(*self.grid_lim_medium, 128)
        self.tst_grid_small = np.linspace(*self.grid_lim_small, 128)
        self.tst_grid_large = np.linspace(*self.grid_lim_large, 128)

        self.tst_coords_medium = np.expand_dims(self.tst_grid_medium, axis=1)
        self.tst_coords_small = np.expand_dims(self.tst_grid_small, axis=1)
        self.tst_coords_large = np.expand_dims(self.tst_grid_large, axis=1)

        self.grid_logp_medium = self.eval_density(self.tst_grid_medium, log=True)
        self.grid_logp_small = self.eval_density(self.tst_grid_small, log=True)
        self.grid_logp_large = self.eval_density(self.tst_grid_large, log=True)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(self.tst_grid_medium)

    def get_density_on_tst_grid(self, tst):
        raise NotImplementedError

    def sample_data(self):
        raise NotImplementedError

    def plot_sequences(self, data, dir_name, name="sequence_of_samples", s=None, label_type=None, gridsize="large"):
        if len(data.shape) == 3:
            n, k, d = data.shape
        else:
            n, d = data.shape
            k = 1
            data = data.reshape(n, k, d)

        grid = getattr(self, "tst_grid_{}".format(gridsize))
        max_n_cols = 4
        n_rows = int(np.ceil(k / max_n_cols))
        n_cols = k if n_rows == 1 else max_n_cols
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axs = axs.ravel() if isinstance(axs, np.ndarray) else [axs]

        for i in range(k):

            x = data[:, i, 0]
            kernel = gaussian_kde(x)
            axs[i].hist(x, bins=int(np.sqrt(x.shape[0])), density=True)
            axs[i].plot(grid, kernel.evaluate(grid))

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

        plt.tight_layout()
        save_fig(dir_name, name)

    def plot_logratios(self, logp_experts, dir_name, fig_name, log_domain=False, gridsize="medium", true_wmarks=None):
        """plot individual experts in the TRE ebm, as well as their sum, on the whole grid

        args
        logp_experts: array of shape (tst_grid_size, n_experts)
        """
        grid = getattr(self, "tst_grid_{}".format(gridsize))

        # plot the true logp versus model logp on
        true_logp = getattr(self, "grid_logp_{}".format(gridsize))
        fig, ax = plt.subplots(1, 1)
        self.plot_product_of_experts(ax, log_domain, logp_experts, true_logp, grid)
        save_fig(dir_name, fig_name + "total", fig=fig)

        n_experts = logp_experts.shape[-1]
        axs, fig = create_subplot_with_max_num_cols(n_experts + 1, max_n_cols=4)
        ax_num = 0

        self.plot_product_of_experts(axs[ax_num], log_domain, logp_experts, true_logp, grid)

        ax_num += 1
        for i in range(ax_num, n_experts+ax_num):
            self.plot_expert(axs[i], i, n_experts + ax_num-1, log_domain, logp_experts, grid, true_wmarks)

        for ax in axs:
            ax.set_xlim(*getattr(self, "grid_lim_{}".format(gridsize)))
            ax.title.set_fontsize(20)

        fig.tight_layout()
        save_fig(dir_name, fig_name, fig=fig)

    def plot_product_of_experts(self, ax, log, logp_experts, true_logp, grid):

        title = r"$ \log \tilde{p}_{model}(\mathbf{x}; \mathbf{\theta})$" if log else\
            r"$ \tilde{p}_{model}(\mathbf{x}; \mathbf{\theta})$"

        total_energy = np.sum(logp_experts, axis=-1)
        e = total_energy if log else np.exp(total_energy)
        true_e = true_logp if log else np.exp(true_logp)
        self.make_plot(ax, grid, e, title, label="model")
        self.make_plot(ax, grid, true_e, label="true")
        self.place_textbox(ax, r"$=$")
        ax.legend()

    def plot_expert(self, ax, i, noise_idx, log_domain, logp_experts, grid, true_wmarks=None):
        j = i-1
        if log_domain:
            title = r"$\log p_{%s}(\mathbf{x})$" % j if i == noise_idx\
                else r"$\log r_{%s}(\mathbf{x}; \mathbf{\theta}_{%s})$" % (j, j)
        else:
            title = r"$p_{%s}(\mathbf{x})$" % j if i == noise_idx \
                else r"$r_{%s}(\mathbf{x}; \mathbf{\theta}_{%s})$" % (j, j)

        # plot estimated ratio (or base distribution)
        r = logp_experts[:, j] if log_domain else np.exp(logp_experts[:, j])
        r[r > 10**32] = 10**32
        self.make_plot(ax, grid, r, title, "model")

        # plot ground truth ratio
        if i < noise_idx and true_wmarks is not None:
            true_r = true_wmarks[j].logpdf(grid) - true_wmarks[j + 1].logpdf(grid)
            true_r = true_r if log_domain else np.exp(true_r)
            true_r[true_r > 10 ** 32] = 10 ** 32
            self.make_plot(ax, grid, true_r, label="true")

        if i < noise_idx:
            self.place_textbox(ax, r"$\plus$ " if log_domain else r"$\times$")

        ax.legend()

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

    def make_plot(self, ax, grid, x, title=None, label=""):
        ax.plot(grid, x, label=label)
        if title:
            ax.set_title(title)
        ax.tick_params(labelsize=14)

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
        return "OneDimData"


class OneDimMoG(OneDimData):
    """
    1d Mixture of Gaussians
    """

    def __init__(self, n_gaussians, mean, std, n_samples, outliers=False, **kwargs):
        self.n_comps = n_gaussians
        self.outliers = outliers
        self.mean = mean
        self.std = std
        self.outer_gauss_coords = [0.2, 0.8]
        self.n_samples = n_samples
        self.make_pdf()
        super().__init__(4*std)

    def eval_density(self, x, log=False):
        if log:
            return logsumexp(np.array([g.logpdf(x) for g in self.gaussians]).T * self.weights, axis=1)
        else:
            return np.sum(np.array([g.pdf(x) for g in self.gaussians]).T * self.weights, axis=1)

    def sample_data(self):
        n_samples_per_gaussian = self.weights * self.n_samples
        samples = np.concatenate([g.rvs(int(n)) for n, g in zip(n_samples_per_gaussian, self.gaussians)], axis=0)
        return samples.reshape(-1, 1)

    def make_pdf(self):
        if self.n_comps == 1:
            centres = np.array([self.mean])
        else:
            centres = np.linspace(*self.outer_gauss_coords, self.n_comps)
            # dist = centres[1] - centres[0]
            # std = dist / 10  # ensures that the gaussians are well-separated

        params = [(c, self.std) for c in centres]
        self.gaussians = [norm(*p) for p in params]
        self.weights = np.ones(self.n_comps) * (1/self.n_comps)

        if self.outliers:
            self.gaussians.append(norm(2.0, self.std))
            self.weights *= 0.9
            self.weights = np.insert(self.weights, self.n_comps, 0.1)

    def __repr__(self):
        return "1d_gauss"


if __name__ == "__main__":
    data_source = OneDimMoG(n_gaussians=1, std=0.01, n_samples=10000)
    fig_dir = os.path.join(project_root, "figs/1d_gauss/")
    os.makedirs(fig_dir, exist_ok=True)
    data_source.plot_sequences(data_source.trn.x, dir_name=fig_dir, name="1d_gauss")
