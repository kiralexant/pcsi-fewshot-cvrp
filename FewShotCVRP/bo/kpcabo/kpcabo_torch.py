import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cma
import numpy as np
import torch
from cma.boundary_handler import BoundTransform
from cma.options_parameters import CMAOptions
from numpy.typing import ArrayLike
from scipy.optimize import Bounds, minimize

import FewShotCVRP.bo.bo_pure as bo_pure
import FewShotCVRP.bo.bo_torch as bo_torch
from FewShotCVRP.bo.bo_pure import _to_2d
from FewShotCVRP.utils.logs import configure_logger, timecall


class KernelPCAMapper:
    """
    Kernel-PCA forward/backward mapper (RBF kernel) for KPCA-BO.

    Forward map uses centered kernel (Schölkopf et al.; also cf. scikit-learn docs):
        g_i(x) = k(x, x_i) - mean_j k(x, x_j) - mean_j k(x_i, x_j) + mean_{i,j} k(x_i, x_j)
        z = g(x) @ V_r    where columns of V_r are normalized eigenvectors of centered Gram

    Backward map (pre-image) follows Antonov et al. (KPCA-BO, Eq.(3)):
        Given z, find x = sum_j w_j p_j, w_j >= 0 (anchors p_j ⊂ training X),
        by minimizing  || z - F(x) ||^2 + Q(x), with Q(x) an exponential box penalty.
    """

    def __init__(
        self,
        bounds: np.ndarray,
        eta: float = 0.9,
        pad_frac: float = 0.1,
        random_state: Optional[int] = None,
        n_restarts_preimage_opt: int = 3,
    ):
        self.bounds = np.asarray(bounds, float)  # (d,2)
        self.eta = float(eta)
        self.pad_frac = float(pad_frac)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.n_restarts_preimage_opt = n_restarts_preimage_opt

        # Fitted attributes
        self.X: np.ndarray = np.empty(
            (0, self.bounds.shape[0]), float
        )  # training X (original space)
        self.Xc: np.ndarray = np.empty(
            (0, self.bounds.shape[0]), float
        )  # centered+weighted X' used for kernel
        self.mu_x: np.ndarray = np.zeros(self.bounds.shape[0], float)

        self.gamma: float = 1.0
        self.K_: np.ndarray = np.empty((0, 0), float)
        self.Kc_: np.ndarray = np.empty((0, 0), float)
        self.K_mean_row_: np.ndarray = np.empty((0,), float)
        self.K_mean_col_: np.ndarray = np.empty((0,), float)
        self.K_mean_all_: float = 0.0

        self.eigvals_: np.ndarray = np.empty((0,), float)
        self.V_: np.ndarray = np.empty((0, 0), float)  # (n, r)
        self.r_: int = 0
        self.z_bounds_: np.ndarray = np.empty((0, 2), float)
        self.logger = configure_logger("FewShotCVRPLogger")

    @staticmethod
    def _rbf(XA: np.ndarray, XB: np.ndarray, gamma: float) -> np.ndarray:
        XA2 = np.sum(XA * XA, axis=1)[:, None]
        XB2 = np.sum(XB * XB, axis=1)[None, :]
        D2 = XA2 + XB2 - 2.0 * (XA @ XB.T)
        return np.exp(-gamma * np.maximum(D2, 0.0))

    @staticmethod
    def _center_kernel(K: np.ndarray):
        n = K.shape[0]
        one = np.ones((n, n), dtype=float) / n
        Kc = K - one @ K - K @ one + one @ K @ one
        mean_row = K.mean(axis=1)
        mean_col = K.mean(axis=0)
        mean_all = float(K.mean())
        return Kc, mean_row, mean_col, mean_all

    def _explained_variance_rank(self, evals: np.ndarray):
        tot = np.sum(evals.clip(min=0.0))
        if tot <= 0:
            return 1, 0.0
        csum = np.cumsum(evals.clip(min=0.0))
        r = int(np.searchsorted(csum, self.eta * tot) + 1)
        r = max(1, min(r, evals.size))
        ratio = csum[r - 1] / tot
        return r, float(ratio)

    def _choose_gamma(self, Xw: np.ndarray) -> float:
        # Coarse-to-fine search for gamma in [1e-4, 2] (log scale), minimizing r - explained_ratio
        gammas = np.logspace(-4, np.log10(2.0), num=16, base=10.0)
        best_gamma = gammas[0]
        best_score = np.inf
        for g in gammas:
            K = self._rbf(Xw, Xw, g)
            Kc, *_ = self._center_kernel(K)
            evals = np.linalg.eigvalsh(Kc)
            evals = np.sort(evals)[::-1]
            r, ratio = self._explained_variance_rank(evals)
            score = r - ratio
            if score < best_score:
                best_score = score
                best_gamma = g
        # local refine
        lg = np.log10(best_gamma)
        neigh = np.logspace(lg - 0.5, lg + 0.5, num=7, base=10.0)
        for g in neigh:
            K = self._rbf(Xw, Xw, g)
            Kc, *_ = self._center_kernel(K)
            evals = np.linalg.eigvalsh(Kc)
            evals = np.sort(evals)[::-1]
            r, ratio = self._explained_variance_rank(evals)
            score = r - ratio
            if score < best_score:
                best_score = score
                best_gamma = g
        return float(best_gamma)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = _to_2d(X)
        n, d = X.shape
        self.X = X.copy()
        self.y = y.copy()
        mu = X.mean(axis=0)
        self.mu_x = mu
        Xc = X - mu[None, :]
        # rank-based weights for minimization: rank 1 is best
        y_argsort = np.argsort(y)
        self.y_sorted = y[y_argsort]
        ranks = np.argsort(y_argsort) + 1  # 1..n
        w_raw = np.log(n) - np.log(ranks)
        self.w_raw_max_rank = w_raw.max()
        w = w_raw / self.w_raw_max_rank
        Xw = Xc * w[:, None]
        self.Xc = Xw

        # choose gamma
        self.gamma = self._choose_gamma(Xw)

        # Gram and centering
        K = self._rbf(Xw, Xw, self.gamma)
        Kc, mean_row, mean_col, mean_all = self._center_kernel(K)
        self.K_ = K
        self.Kc_ = Kc
        self.K_mean_row_ = mean_row
        self.K_mean_col_ = mean_col
        self.K_mean_all_ = mean_all

        # eigen-decomposition
        evals, evecs = np.linalg.eigh(Kc)  # ascending
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

        r, _ = self._explained_variance_rank(evals)
        self.r_ = int(max(1, r))

        evecs_r = evecs[:, : self.r_]
        evals_r = np.clip(evals[: self.r_], 1e-12, None)
        V = evecs_r / np.sqrt(evals_r[None, :])  # normalized eigenvectors
        self.V_ = V
        self.eigvals_ = evals

        # z-bounds from current training images with padding
        Z = self.transform(self.X, self.y)
        zmin = Z.min(axis=0)
        zmax = Z.max(axis=0)
        pad = self.pad_frac * (zmax - zmin + 1e-12)
        self.z_bounds_ = np.vstack([zmin - pad, zmax + pad]).T

    @staticmethod
    def nearest_index(y_sorted: np.ndarray, y_new: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(y_new).astype(y_sorted.dtype)
        pos = np.searchsorted(y_sorted, x, side="left")  # insertion positions

        # neighbor candidates (left and right)
        left = np.clip(pos - 1, 0, len(y_sorted) - 1)
        right = np.clip(pos, 0, len(y_sorted) - 1)

        # pick nearest (on ties, choose left/lower)
        choose_left = (pos > 0) & (
            (pos == len(y_sorted))
            | (np.abs(x - y_sorted[left]) <= np.abs(y_sorted[right] - x))
        )
        return np.where(choose_left, left, right)

    def _rescale_Xnew(self, Xnew: np.ndarray, ynew: np.ndarray) -> np.ndarray:
        Xnew = _to_2d(Xnew)
        Xnew_c = Xnew - self.mu_x[None, :]
        ranks_new = self.nearest_index(y_sorted=self.y_sorted, y_new=ynew) + 1
        n, _ = self.X.shape
        w_raw = np.log(n) - np.log(ranks_new)
        w = w_raw / self.w_raw_max_rank
        Xnew_w = Xnew_c * w[:, None]
        return Xnew_w

    def _g_vec(self, Xnew: np.ndarray, ynew: np.ndarray) -> np.ndarray:
        Xnew_w = self._rescale_Xnew(Xnew, ynew)
        Kx = self._rbf(Xnew_w, self.Xc, self.gamma)  # (m,n)
        mean_over_cols = Kx.mean(axis=1)
        g = Kx - mean_over_cols[:, None] - self.K_mean_row_[None, :] + self.K_mean_all_
        return g

    def transform(self, Xnew: np.ndarray, ynew: np.ndarray) -> np.ndarray:
        g = self._g_vec(Xnew, ynew)
        Z = g @ self.V_
        return Z

    def _penalty_Q(self, x: np.ndarray) -> float:
        l = self.bounds[:, 0]
        u = self.bounds[:, 1]
        over = np.maximum(0.0, x - u)
        under = np.maximum(0.0, l - x)
        return float(np.exp(np.sum(over + under))) - 1.0

    def _opt_with_linear_comb(
        self, P, z, z_obj_value_approx, m, max_transforms, plotting_callback
    ):
        def x_from_w(w):
            # w = np.maximum(w, 0.0)
            return (P.T @ w).T  # (d,)

        def obj(w):
            x = x_from_w(w)
            z_hat = self.transform(x[None, :], z_obj_value_approx)[0]
            diff = z_hat - z
            dist_closest = np.min(np.sum((P - x) ** 2, axis=1))
            f = float(np.dot(diff, diff) + self._penalty_Q(x)) + 0.01 * dist_closest
            return f

        def obj_grad(w):
            f = obj(w)
            eps = 1e-6
            g = np.zeros_like(w)
            for i in range(m):
                wi = w[i]
                w[i] = wi + eps
                f2 = obj(w)
                g[i] = (f2 - f) / eps
                w[i] = wi
            return f, g

        # w0 = np.zeros(m, dtype=float)
        lo = np.ones(m) * -1
        up = np.ones(m)
        remaining_budget = max_transforms
        w_star, res_best, f_star = None, None, np.inf
        for i in range(m):
            x0 = np.zeros(m)
            x0[i] = 1

            sigma0 = np.mean(up - lo) / 10
            tolx = 1e-10
            tolfun = 1e-10
            tolfunhist = 1e-10
            base_opts = {
                "integer_variables": None,
                "BoundaryHandler": BoundTransform,
                "bounds": (lo, up),
                "maxfevals": remaining_budget,
                "tolx": tolx,
                "tolfun": tolfun,
                "tolfunhist": tolfunhist,
                "verb_disp": 0,
                "verb_log": 0,
                "seed": self.random_state,
            }

            opts = CMAOptions(base_opts)
            # x0 = self.rng.uniform(low=lo, high=up, size=m)
            # x0 = (up + lo) / 2
            # x0 = np.array([1.0, 0.0])
            es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
            res = es.optimize(obj).result
            remaining_budget -= res.evaluations
            # print(res.fbest)
            if res.fbest < f_star:
                w_star, res_best, f_star = res.xbest, res, res.fbest

        if plotting_callback is not None:
            plotting_callback(obj, res_best)

        x_star = x_from_w(w_star)
        return x_star

    @staticmethod
    def _project_to_simplex(y):
        """Euclidean projection onto the probability simplex.
        Based on Wang & Carreira-Perpiñán (2013); O(d log d).

        Wang, Weiran, and Miguel A. Carreira-Perpinán.
        "Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application."
        arXiv preprint arXiv:1309.1541 (2013).
        """
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        d = y.size
        u = np.sort(y)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, d + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1.0) / (rho + 1)
        x = np.maximum(y - theta, 0.0)
        return x

    @staticmethod
    def _penalty_distance_to_simplex(w):
        w = np.asarray(w, dtype=float)
        wp = KernelPCAMapper._project_to_simplex(w)
        return np.sum((w - wp) ** 2), wp

    @staticmethod
    def wrapped_objective_for_cma_es(w, f, rho=1.0):
        """Repair+penalty wrapper: evaluate f on the projected point and add distance penalty."""
        pen, wp = KernelPCAMapper._penalty_distance_to_simplex(w)
        return float(f(wp) + rho * pen)

    def _opt_with_convex_comb(
        self, P, z, z_obj_value_approx, m, max_transforms, plotting_callback
    ):
        def x_from_w(w):
            return (P.T @ w).T  # (d,)

        def obj(w):
            pen, wp = KernelPCAMapper._penalty_distance_to_simplex(w)
            x = x_from_w(wp)
            z_hat = self.transform(x[None, :], z_obj_value_approx)[0]
            diff = z_hat - z
            # dist_closest = np.min(np.sum((P - x) ** 2, axis=1))
            f = float(np.dot(diff, diff) + self._penalty_Q(x)) + pen
            return f

        def log_obj(w):
            return max(-100.0, np.log(obj(w)))

        def obj_grad_post(w):
            f = np.log(obj(w))
            g = np.zeros_like(w)
            eps = 1e-10
            for i in range(m):
                wi = w[i]
                w[i] = wi + eps
                f2 = np.log(obj(w))
                g[i] = (f2 - f) / eps
                w[i] = wi
            return f, g

        lo = np.zeros(m) - 0.1
        up = np.ones(m) + 0.1
        remaining_budget = max_transforms
        w_star, res_best, f_star = None, None, np.inf
        for i in range(self.n_restarts_preimage_opt):
            if remaining_budget <= 0:
                break
            if i == 0:
                x0 = self.rng.uniform(low=lo, high=up, size=m)

                sigma0 = np.mean(up - lo) / 6
                tolx = 1e-10
                tolfun = 1e-10
                tolfunhist = 1e-10
                base_opts = {
                    "integer_variables": None,
                    "BoundaryHandler": BoundTransform,
                    "bounds": (lo, up),
                    "maxfevals": remaining_budget,
                    "tolx": tolx,
                    "tolfun": tolfun,
                    "tolfunhist": tolfunhist,
                    "verb_disp": 0,
                    "verb_log": 0,
                    "seed": self.random_state,
                }
                opts = CMAOptions(base_opts)
                es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
                res = es.optimize(obj).result
                res_fbest = res.fbest
            else:
                x0 = w_star.copy()
                sigma0 = np.mean(up - lo) / 12

                tolx = 1e-10
                tolfun = 1e-3
                tolfunhist = 1e-1
                base_opts = {
                    "integer_variables": None,
                    "BoundaryHandler": BoundTransform,
                    "bounds": (lo, up),
                    "maxfevals": remaining_budget,
                    "tolx": tolx,
                    "tolfun": tolfun,
                    "tolfunhist": tolfunhist,
                    "verb_disp": 0,
                    "verb_log": 0,
                    "seed": self.random_state,
                }
                opts = CMAOptions(base_opts)
                es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
                res = es.optimize(log_obj).result
                res_fbest = obj(res.xbest)

            remaining_budget -= res.evaluations

            if res_fbest < f_star:
                w_star, res_best, f_star = res.xbest, res, res_fbest

        if plotting_callback is not None:
            plotting_callback(obj, res_best)

        self.logger.info(f"Preimage location final quality: {f_star:6f}")
        x_star = x_from_w(w_star)
        return x_star

    def _opt_with_convex_comb_with_post_grad(
        self, P, z, z_obj_value_approx, m, max_transforms, plotting_callback
    ):
        def x_from_w(w):
            # w = np.maximum(w, 0.0)
            return (P.T @ w).T  # (d,)

        def obj(w):
            pen, wp = KernelPCAMapper._penalty_distance_to_simplex(w)
            x = x_from_w(wp)
            z_hat = self.transform(x[None, :], z_obj_value_approx)[0]
            diff = z_hat - z
            # dist_closest = np.min(np.sum((P - x) ** 2, axis=1))
            f = float(np.dot(diff, diff) + self._penalty_Q(x)) + pen
            return f

        def obj_grad_post_old(w):
            x = x_from_w(w)
            z_hat = self.transform(x[None, :], z_obj_value_approx)[0]
            diff = z_hat - z
            f = np.log(np.dot(diff, diff))
            g = np.zeros_like(w)
            eps = 1e-6
            for i in range(m):
                wi = w[i]
                w[i] = wi + eps
                x2 = x_from_w(w)
                z2_hat = self.transform(x2[None, :], z_obj_value_approx)[0]
                diff2 = z2_hat - z
                f2 = np.log(np.dot(diff2, diff2))
                g[i] = (f2 - f) / eps
                w[i] = wi
            return f, g

        def obj_grad_post(w):
            f = np.log(obj(w))
            g = np.zeros_like(w)
            eps = 1e-10
            for i in range(m):
                wi = w[i]
                w[i] = wi + eps
                f2 = np.log(obj(w))
                g[i] = (f2 - f) / eps
                w[i] = wi
            return f, g

        lo = np.zeros(m) - 0.1
        up = np.ones(m) + 0.1
        remaining_budget = max_transforms
        w_star, res_best, f_star = None, None, np.inf
        for i in range(self.n_restarts_preimage_opt):
            # x0 = np.zeros(m)
            # x0[i] = 1
            sigma0 = np.mean(up - lo) / 6
            if w_star is None:
                x0 = self.rng.uniform(low=lo, high=up, size=m)
            else:
                x0 = w_star.copy()
                sigma0 *= 2

            tolx = 1e-10
            tolfun = 1e-10
            tolfunhist = 1e-10
            base_opts = {
                "integer_variables": None,
                "BoundaryHandler": BoundTransform,
                "bounds": (lo, up),
                "maxfevals": remaining_budget,
                "tolx": tolx,
                "tolfun": tolfun,
                "tolfunhist": tolfunhist,
                "verb_disp": 0,
                "verb_log": 0,
                "seed": self.random_state,
            }

            opts = CMAOptions(base_opts)
            # x0 = self.rng.uniform(low=lo, high=up, size=m)
            # x0 = (up + lo) / 2
            # x0 = np.array([1.0, 0.0])
            es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
            res = es.optimize(obj).result
            remaining_budget -= res.evaluations

            # if res.fbest < f_star:
            #     w_star, res_best, f_star = res.xbest, res, res.fbest

            res1 = minimize(
                obj_grad_post,
                res.xbest,
                method="L-BFGS-B",
                bounds=np.array(list(zip(lo, up))),
                options={"maxiter": 100, "ftol": 1e-3},
                jac=True,
            )

            res1_fun = obj(res1.x)
            # print(res.fbest)
            if res1_fun < f_star:
                w_star, res_best, f_star = res1.x, res1, res1_fun

            # print(res.fbest, res1_fun)

        if plotting_callback is not None:
            plotting_callback(obj, res_best)

        x_star = x_from_w(w_star)
        return x_star

    def preimage(
        self,
        z: np.ndarray,
        z_obj_value_approx: np.ndarray,
        anchors: Optional[np.ndarray] = None,
        max_anchors: int = 64,
        max_transforms: int = 200,
        opt_strategy: str = "convex comb",
        plotting_callback: Any = None,
    ) -> np.ndarray:
        z = np.asarray(z, float).reshape(-1)
        z_obj_value_approx = np.atleast_1d(z_obj_value_approx)
        n = self.X.shape[0]
        Z_train = self.transform(self.X, self.y)
        if anchors is None:
            d2 = np.sum((Z_train - z[None, :]) ** 2, axis=1)
            idx = np.argsort(d2)[: min(max_anchors, n)]
            # print(*idx, sep=",")
            P = self.X[idx]
        else:
            P = np.asarray(anchors, float)
        m = P.shape[0]

        if opt_strategy == "linear comb":
            x_star = self._opt_with_linear_comb(
                P, z, z_obj_value_approx, m, max_transforms, plotting_callback
            )
        elif opt_strategy == "convex comb":
            x_star = self._opt_with_convex_comb(
                P, z, z_obj_value_approx, m, max_transforms, plotting_callback
            )
        elif opt_strategy == "convex comb with post grad":
            x_star = self._opt_with_convex_comb_with_post_grad(
                P, z, z_obj_value_approx, m, max_transforms, plotting_callback
            )

        l, u = self.bounds[:, 0], self.bounds[:, 1]
        return np.clip(x_star, l, u)

    def z_bounds(self) -> np.ndarray:
        return self.z_bounds_.copy()


@dataclass
class KPCABayesianOptimizer:
    """
    Kernel-PCA-BO: Bayesian Optimization in a KPCA-reduced subspace.
    Extends the lean `bo_pure.BayesianOptimizer` but runs acquisition in z-space.
    """

    # Objective(s)
    func: Callable[[np.ndarray], float]
    f_batch: Optional[Callable[[np.ndarray], ArrayLike]] = None

    # Embedding control
    kpca_eta: float = 0.90
    kpca_pad_frac: float = 0.10
    kpca_refit_trigger: float = 0.20
    max_preimage_anchors: int = 3
    preimage_budget: int = 5000

    # Original space
    bounds: np.ndarray = None
    minimize: bool = True

    # BO in embedding space
    bo_embedding_kwargs: dict = None
    n_init: int = 50
    n_iter: int = 15
    suggestions_per_step: int = 10

    # Device / dtype
    device: Union[str, torch.device] = "cpu"
    dtype: torch.dtype = torch.double

    # Reproducibility
    random_state: Optional[int] = None

    # Runtime
    X_np_: np.ndarray = field(init=False, repr=False)  # (n,d)
    Z_np_: np.ndarray = field(init=False, repr=False)  # (n,d)
    y_np_: np.ndarray = field(init=False, repr=False)  # (n,1)
    gp_: Any = field(init=False, repr=False)
    logger: Any = field(init=False, repr=False)

    def __post_init__(self):
        self.bo_embedding_kwargs = self.bo_embedding_kwargs.copy()
        self.bo_embedding_kwargs.pop("func", None)
        self.bo_embedding_kwargs.pop("f_batch", None)
        self.bo_embedding_kwargs.update({"device": self.device})
        self.bo_embedding_kwargs.update({"dtype": self.dtype})
        if self.bo_embedding_kwargs is None:
            self.bo_embedding_kwargs = {}
        self.logger = configure_logger("FewShotCVRPLogger")
        self._find_preimages = timecall(self.logger, "find_preimages")(
            self._find_preimages
        )
        self._fit_kpca = timecall(self.logger, "Refitting Kernel-PCA done! fit_kpca")(
            self._fit_kpca
        )
        self._evaluate_objective = timecall(self.logger, "evaluate_objective")(
            self._evaluate_objective
        )

    def _fit_kpca(self):
        self.logger.info("Refitting Kernel-PCA triggered")
        self.mapper_ = KernelPCAMapper(
            bounds=self.bounds,
            eta=self.kpca_eta,
            pad_frac=self.kpca_pad_frac,
            random_state=self.random_state,
            n_restarts_preimage_opt=self.max_preimage_anchors,
        )
        self.mapper_.fit(self.X_np_, self.y_np_.squeeze(-1))
        self.Z_np_ = self.mapper_.transform(self.X_np_, self.y_np_.squeeze(-1))
        self.logger.info(
            f"Dimensionality of the embedding space: {self.Z_np_.shape[-1]}"
        )

    def _should_refit_kpca(self, y_new: np.ndarray) -> bool:
        y_all = np.concatenate([self.y_np_, y_new])
        thr = np.quantile(y_all, self.kpca_refit_trigger)
        return np.min(y_new) <= thr

    def _create_bo_embedding(self):
        z_bounds = self.mapper_.z_bounds()
        self.bo_embedding_kwargs.update({"bounds": z_bounds})
        self.bo_embedding_kwargs.update({"X_init": self.Z_np_})
        self.bo_embedding_kwargs.update({"y_init": self.y_np_})
        self.bo_embedding_kwargs.update({"doe_method": "precomputed"})
        bo_embedding = bo_torch.BayesianOptimizer(
            func=self.func, f_batch=self.f_batch, **self.bo_embedding_kwargs
        )
        bo_embedding._initialize_design()
        return bo_embedding

    def _find_preimages(self, Z, values):
        return np.vstack(
            [
                self.mapper_.preimage(
                    z,
                    z_value,
                    max_anchors=self.max_preimage_anchors,
                    max_transforms=self.preimage_budget,
                )
                for z, z_value in zip(Z, values)
            ]
        )

    def _evaluate_objective(self, bo_embedding, X_new_np):
        return bo_embedding._evaluate_objective_np(X_new_np).reshape(-1, 1)

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        # 1) Refit GP in z-space for next round
        bo_embedding = self._create_bo_embedding()
        bo_embedding._fit_gp()
        self.gp_ = bo_embedding.get_gp()

        # 2) Optimize EI in z-space
        Z_new_t, _ = bo_embedding._argmax_acq_multi(bo_embedding.suggestions_per_step)
        Z_new_obj_values_approx, _ = bo_embedding.posterior(Z_new_t, return_std=False)
        Z_new = bo_torch._to_numpy64(Z_new_t)

        # 3) Back-map to x-space
        X_new_np = self._find_preimages(Z_new, Z_new_obj_values_approx)

        # 4) Evaluate in x-space
        y_new_np = self._evaluate_objective(bo_embedding, X_new_np)

        # 5) Update data & mapper (refit or append)
        self.X_np_ = np.vstack([self.X_np_, X_new_np])
        self.y_np_ = np.concatenate([self.y_np_, y_new_np])

        self._fit_kpca()
        # if self._should_refit_kpca(y_new_np):
        #     self._fit_kpca()
        # else:
        #     self.Z_np_ = np.vstack(
        #         [self.Z_np_, self.mapper_.transform(X_new_np, Z_new_obj_values_approx)]
        #     )

    def run(self) -> Dict[str, np.ndarray]:
        self.bo_embedding_kwargs.update({"bounds": self.bounds})
        self.bo_embedding_kwargs.update({"n_init": self.n_init})
        bo_embedding = bo_torch.BayesianOptimizer(
            func=self.func, f_batch=self.f_batch, **self.bo_embedding_kwargs
        )
        bo_embedding._initialize_design()
        self.X_np_ = bo_torch._to_numpy64(bo_embedding.X_)
        self.y_np_ = bo_torch._to_numpy64(bo_embedding.y_)
        self._fit_kpca()
        for _ in range(self.n_iter):
            self.step()

        bo_embedding = self._create_bo_embedding()
        bo_embedding._fit_gp()
        self.gp_ = bo_embedding.get_gp()

        x_obs, y_obs = self.recommend(mode="observed", refit_gp=False)

        x_mean_in_data, y_mean_in_data = self.recommend(
            mode="mean_in_data", bo_embedding=bo_embedding, refit_gp=False
        )
        x_rec_mean_global, y_rec_mean_global = self.recommend(
            mode="mean_global", bo_embedding=bo_embedding, refit_gp=False
        )

        return {
            "x_obs_best": x_obs,
            "y_obs_best": y_obs,
            "x_rec_mean_in_data": x_mean_in_data,
            "y_rec_mean_in_data": y_mean_in_data,
            "x_rec_mean_global": x_rec_mean_global,
            "y_rec_mean_global": y_rec_mean_global,
            "X": self.X_np_,
            "y": self.y_np_.squeeze(-1),
        }

    def recommend(
        self,
        bo_embedding: bo_torch.BayesianOptimizer = None,
        mode: str = "mean_global",
        n_restarts_opt_mean_global: int = 100,
        refit_gp: bool = False,
    ):
        if mode == "observed":
            y_flat = self.y_np_.squeeze(-1)
            i = int(np.argmin(y_flat) if self.minimize else torch.argmax(y_flat))
            return self.X_np_[i], float(y_flat[i])

        if mode == "mean_in_data" or mode == "mean_global":
            z_rec_mean_in_data, y_rec_mean_in_data = bo_embedding.recommend(
                mode=mode,
                refit_gp=refit_gp,
                n_restarts_opt_mean_global=n_restarts_opt_mean_global,
            )
            x_rec_mean_in_data = self.mapper_.preimage(
                z_rec_mean_in_data,
                y_rec_mean_in_data,
                max_anchors=self.max_preimage_anchors,
                max_transforms=self.preimage_budget,
            )
            return x_rec_mean_in_data, y_rec_mean_in_data

    # ---------------- snapshot ----------------
    def save_snapshot(self, save_dir: str, results: Dict = None) -> Path:
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)

        # Save model params
        if hasattr(self, "gp_"):
            torch.save(self.gp_.state_dict(), d / "gp_state.pt")

        arrays = {
            "X_": self.X_np_,
            "y_": self.y_np_.squeeze(-1),
            "Z_": self.Z_np_,
            "bounds": np.asarray(self.bounds, dtype=np.float64),
        }
        np.savez_compressed(d / "arrays.npz", **arrays)

        manifest = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "gpytorch_version": __import__("gpytorch").__version__,
            "botorch_version": __import__("botorch").__version__,
        }
        (d / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        if results is None:
            y = arrays["y_"]
            X = arrays["X_"]
            Z = arrays["Z_"]
            if y.size:
                idx = int(np.argmin(y) if self.minimize else np.argmax(y))
                x_best = X[idx].tolist()
                y_best = float(y[idx])
            else:
                x_best, y_best = None, None
            result_json = {
                "x_obs_best": x_best,
                "y_obs_best": y_best,
                "X_shape": list(X.shape),
                "Z_shape": list(Z.shape),
                "y_shape": list(y.shape),
            }
        else:
            result_json = {
                "x_obs_best": results["x_obs_best"].tolist(),
                "y_obs_best": results["y_obs_best"],
                "x_rec_mean_in_data": results["x_rec_mean_in_data"].tolist(),
                "y_rec_mean_in_data": results["y_rec_mean_in_data"],
                "x_rec_mean_global": results["x_rec_mean_global"].tolist(),
                "y_rec_mean_global": results["y_rec_mean_global"],
                "X_shape": list(arrays["X_"].shape),
                "y_shape": list(arrays["y_"].shape),
                "Z_shape": list(arrays["Z_"].shape),
            }
        (d / "result.json").write_text(
            json.dumps(result_json, indent=2), encoding="utf-8"
        )
        return d

    @staticmethod
    def load_snapshot(load_dir: str) -> bo_torch.BOSnapshot:
        d = Path(load_dir)
        gp_state = torch.load(d / "gp_state.pt", map_location="cpu")
        arrays_file = np.load(d / "arrays.npz")
        arrays = {k: arrays_file[k] for k in arrays_file.files}
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        result = json.loads((d / "result.json").read_text(encoding="utf-8"))
        experiment_config = None
        if (d / "experiment-config.json").is_file():
            experiment_config = json.loads(
                (d / "experiment-config.json").read_text(encoding="utf-8")
            )
        return bo_torch.BOSnapshot(
            gp=gp_state,
            arrays=arrays,
            manifest=manifest,
            result=result,
            experiment_config=experiment_config,
        )
