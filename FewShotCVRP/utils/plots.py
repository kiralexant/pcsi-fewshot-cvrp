from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import root_scalar
from scipy.stats import special_ortho_group


def good_plt_config():
    plt.style.use("default")
    script_dir = Path(__file__).resolve().parent
    with open(script_dir / "latex-preambula.tex", "r") as f:
        latex_preambula = f.read()
    plt.rcParams["text.usetex"] = True
    plt.rc("text.latex", preamble=latex_preambula)
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.linestyle"] = (0, (5, 5))
    plt.rcParams["grid.linewidth"] = 0.5
    mpl.rcParams["font.size"] = 15
    plt.rcParams["xtick.labelsize"] = 15
    plt.rcParams["ytick.labelsize"] = 15


def default_plt_config():
    plt.style.use("default")
    sns.set(style="whitegrid", context="talk")


def compose_points(f, x, y):
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(X)):
            Z[i][j] = f(np.array([X[i][j], Y[i][j]]))
    return X, Y, Z


def compute_zax_min_max(zax_min, zax_max, zfactor=1.0, inversion=False):
    h = zax_max - zax_min
    shift = 2 * h * zfactor
    if inversion:
        return zax_max + shift, zax_min
    return zax_min - shift, zax_max


def rastrigin(x):
    N = len(x)
    # c = 2*math.pi
    c = 1.5
    return 10 * N + sum(x[i] ** 2 - 10 * np.cos(c * x[i]) for i in range(N))


def sphere(x):
    return sum((xi - 1) ** 2 for xi in x)


def truncated_exponential_mean(lambda_, b):
    numerator = 1 - np.exp(-lambda_ * b) * (1 + lambda_ * b)
    denominator = lambda_ * (1 - np.exp(-lambda_ * b))
    return numerator / denominator


def find_lambda_for_mean(mean, b):
    def objective(lambda_):
        return truncated_exponential_mean(lambda_, b) - mean

    sol = root_scalar(objective, bracket=[1e-6, 1e6], method="brentq")
    return sol.root


class GallagherFunction:
    def __init__(
        self,
        num_peaks: int = 10,
        dim: int = 2,
        min_peak_distance: float = 2.0,
        seed: int | None = None,
        max_weight: float = 100.0,
        min_cond_number: float = 1.0,
        mean_cond_number: float = 5.0,
        max_cond_number: float = 1e3,
        min_dist: float = 1.0,
        max_dist: float = 2.0,
    ):
        """
        Initialize parameters for the Gallagher multimodal test function.

        Parameters
        ----------
        num_peaks : int, optional
            Number of local optima (“peaks”) in the search space. Default is 10.
        dim : int, optional
            Dimensionality of the search space (number of variables). Default is 2.
        min_peak_distance : float, optional
            Minimum Euclidean distance between any two peaks to ensure they
            are sufficiently separated. Default is 2.0.
        seed : int or None, optional
            Seed for the random number generator to ensure reproducibility.
            If None, a random seed is used. Default is None.
        max_weight : float, optional
            Maximum “height” of a peak -- i.e., the absolute objective value of
            the smallest local minimum. Controls the contrast between optima.
            Default is 100.0.
        min_cond_number : float, optional
            Minimum condition number of the ellipsoidal attraction basins
            around the optima, determining how “stretched” each basin is.
            Default is 1.0.
        mean_cond_number : float, optional
            Mean condition number of all basins, used when sampling the
            distribution of condition numbers. Default is 5.0.
        max_cond_number : float, optional
            Maximum condition number for any basin. Default is 1e3.
        min_dist : float, optional
            Minimum radius from a basin's center at which the function value
            reaches -1, controlling basin width. Default is 1.0.
        max_dist : float, optional
            Maximum such radius, setting an upper bound on basin width.
            Default is 2.0.

        Notes
        -----
        - Varying the condition numbers allows modeling basins from nearly
          spherical to highly elongated shapes.
        - The min_dist and max_dist parameters guarantee the function
          attains the value -1 within the specified radius of each peak.
        """
        self.num_peaks = num_peaks
        self.dim = dim
        self.min_peak_distance = min_peak_distance
        self.rng = np.random.default_rng(seed)
        self.max_weight = max_weight
        self.min_cond_number = min_cond_number
        self.mean_cond_number = mean_cond_number
        self.max_cond_number = max_cond_number
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.__build_function()

    def __build_function(self):
        # generate peaks without overlap
        self.peaks = []
        while len(self.peaks) < self.num_peaks:
            candidate = self.rng.uniform(-4.5, 4.5, size=self.dim)
            if all(
                np.linalg.norm(candidate - p) >= self.min_peak_distance
                for p in self.peaks
            ):
                self.peaks.append(candidate)
        self.peaks = np.array(self.peaks)

        lam = find_lambda_for_mean(self.mean_cond_number, self.max_cond_number)
        sample_trunc_exp = (
            lambda lam: self.min_cond_number
            - np.log(
                1
                - self.rng.uniform(size=1)[0]
                * (1 - np.exp(-lam * self.max_cond_number))
            )
            / lam
        )

        self.weight = np.ones(self.num_peaks) * self.max_weight
        self.covs = []
        self.cond_numbers = []
        self.max_eigvalues = []
        self.min_eigvalues = []

        for i in range(self.num_peaks):
            cond_number = sample_trunc_exp(lam)
            self.cond_numbers.append(cond_number)
            dist = self.rng.uniform(self.min_dist, self.max_dist)
            max_eigv = -2 * np.log(1 / self.weight[i]) / dist
            min_eigv = max_eigv / cond_number
            self.max_eigvalues.append(max_eigv)
            self.min_eigvalues.append(min_eigv)
            eigvals = np.concatenate(
                (
                    np.array([max_eigv, min_eigv]),
                    self.rng.uniform(min_eigv, max_eigv, size=self.dim - 2),
                )
            )
            R = special_ortho_group.rvs(self.dim, random_state=self.rng)
            self.covs.append(R.T @ np.diag(eigvals) @ R)

    def __call__(self, x: np.ndarray):
        ans = 0
        for i, x_star in enumerate(self.peaks):
            qf = np.exp(-0.5 * (x - x_star).T @ self.covs[i] @ (x - x_star))
            ans = max(
                ans,
                self.weight[i] * qf,
            )
        return -ans


def plot_3D_surface(
    x1_lims,
    x2_lims,
    f,
    X_data,
    discretization=50,
    is_colorbar=False,
    is_axis_names=False,
    is_white_facecolor=False,
    is_scatter_search_space=True,
    scatter_search_space_color="red",
    is_scatter_objective_space=False,
    zfactor=1.0,
    is_inverse=False,
    is_remove_extra_zlables=False,
    zlabelpad=10,
    is_connect_search_obj=False,
):
    x1 = np.linspace(*x1_lims, discretization)
    x2 = np.linspace(*x2_lims, discretization)
    X1, X2, Z = compose_points(f, x1, x2)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d", "computed_zorder": False})
    fig.set_size_inches(18.5, 10.5)
    mycmap = mpl.cm.jet
    ax.plot_surface(
        X1,
        X2,
        Z,
        cmap=mycmap,
        antialiased=True,
        linewidth=0.2,
        edgecolor="k",
        rcount=len(X1),
        ccount=len(X1[0]),
        alpha=1,
        zorder=3,
    )
    zax_min, zax_max = ax.get_zlim()
    zax_min, zax_max = compute_zax_min_max(zax_min, zax_max, zfactor, is_inverse)
    ax.set_zlim(zax_min, zax_max)

    if is_remove_extra_zlables:
        ticks = ax.get_zticks()
        labels = ax.get_zticklabels()
        if is_inverse:
            is_prv = False
            _zmax = Z.max()
            for tick_val, label in zip(ticks, labels):
                if is_prv:
                    label.set_visible(False)
                if tick_val > _zmax:
                    is_prv = True
        else:
            _zmin = 1.1 * Z.min()
            for tick_val, label in zip(ticks, labels):
                if tick_val < _zmin:
                    label.set_visible(False)

    ax.contourf(
        X1,
        X2,
        Z,
        zdir="z",
        offset=zax_min,
        cmap=mycmap,
        extend="both",
        levels=50,
        alpha=0.4,
        zorder=2,
    )
    ax.contour(
        X1,
        X2,
        Z,
        zdir="z",
        offset=zax_min,
        cmap=mycmap,
        # colors='k',
        extend="both",
        levels=50,
        alpha=1,
        zorder=2,
        linewidths=0.5,
        # linestyles='--',
    )
    if len(X_data) > 0:
        y_data = [f(x) for x in X_data]
        if is_scatter_objective_space:
            ax.scatter(
                X_data[:, 0],
                X_data[:, 1],
                y_data,
                c="magenta",
                marker="+",
                s=100,
                alpha=1,
                zorder=4,
            )
        if is_scatter_search_space:
            ax.scatter(
                X_data[:, 0],
                X_data[:, 1],
                zax_min,
                c=scatter_search_space_color,
                marker="+",
                s=100,
                alpha=1,
                zorder=4,
            )
        if (
            is_scatter_search_space
            and is_scatter_search_space
            and is_connect_search_obj
        ):
            for i, x in enumerate(X_data):
                ax.plot(
                    [x[0], x[0]],  # X stays constant
                    [x[1], x[1]],  # Y stays constant
                    [zax_min, y_data[i]],  # Z goes from z1 to z2
                    color="k",
                    linestyle="--",
                    linewidth=1,
                    zorder=2,
                )
    if is_colorbar:
        zmin, zmax = Z.min(), Z.max()
        sm = plt.cm.ScalarMappable(cmap=mycmap, norm=plt.Normalize(zmin, zmax))
        fig.colorbar(
            sm,
            ax=ax,
            shrink=0.7,
            ticks=np.linspace(zmin, zmax, 5),
            orientation="vertical",
            extend="both",
        )
    if is_axis_names:
        ax.set_xlabel(r"$x_1$", labelpad=10)
        ax.set_ylabel(r"$x_2$", labelpad=10)
        zlabel = r"$f\!\br{\bm{x}}$"
        if is_inverse:
            zlabel = r"$\text{\textbf{\textcolor{red}{(inversed)}}}$ " + zlabel
        ax.set_zlabel(zlabel, labelpad=zlabelpad)
    if is_white_facecolor:
        ax.get_xaxis().set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.get_yaxis().set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.get_zaxis().set_pane_color((1.0, 1.0, 1.0, 1.0))
    return fig, ax
