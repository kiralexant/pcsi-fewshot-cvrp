# bo_pure.py -- a Bayesian Optimization implementation without unnecessary code.
#
# Features
# - Single-point EI (Expected Improvement) with analytic gradient (fast L-BFGS-B).
# - scikit-learn Gaussian Process with Constant * (Matern(ν=2.5) | RBF) [+ optional White].
# - Known observation noise via `sigma` (std): uses alpha = sigma**2; otherwise WhiteKernel is fitted.
# - Access final GP via .get_gp().
# - ARD helpers: .get_length_scales() and .report_ard(...).
# - Batch-friendly API:
#     * Optional `f_batch(X: np.ndarray) -> ArrayLike` to evaluate many points at once
#       during initialization and BO steps.
#     * Multi-suggestion per step via `suggestions_per_step` (q >= 1) using
#       multiple local maximizers of EI with diversity filtering.
#
# Dependencies:
#     pip install numpy scipy scikit-learn

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import sklearn
from numpy.typing import ArrayLike
from scipy.linalg import cho_solve
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


def _to_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X


@dataclass
class BOSnapshot:
    gp: GaussianProcessRegressor
    arrays: Dict[str, np.ndarray]  # all np-atributes (X_, y_, bounds, …)
    manifest: Dict[str, Any]  # versions/date
    result: Dict[str, Any]  # x_best / y_best / shapes


@dataclass
class BayesianOptimizer:
    # Objective APIs
    func: Callable[[np.ndarray], float]
    f_batch: Optional[Callable[[np.ndarray], ArrayLike]] = (
        None  # vectorized evaluator (n,d)->(n,)
    )

    # Initial set of points (DoE)
    X_init: Optional[np.ndarray] = None
    y_init: Optional[np.ndarray] = None

    # Domain and BO config
    bounds: Sequence[Tuple[float, float]] = ()
    n_init: int = 5
    n_iter: int = 25
    minimize: bool = True
    xi: float = 0.01  # exploration parameter for EI

    # Noise
    sigma: Optional[Union[float, np.ndarray]] = (
        None  # observation noise std (scalar or per-sample)
    )
    noise_std_guess: Optional[float] = (
        None  # automatically fit the observation noise std if sigma is not set
    )
    noise_std_bounds: Optional[Tuple[float, float]] = (
        None  # bounds for the fitted noise
    )

    # GP kernel
    kernel: str = "matern"  # "matern" or "rbf"
    kernel_isotropic: bool = False  # Kernel с одним length_scale на все фичи
    length_scale_bounds: Optional[np.ndarray] = (
        None  # bounds for the lengh_scale used in RBF or Matern
    )
    c_factor_bounds: Optional[np.ndarray] = (
        None  # bounds for the constant factor of the kernel
    )

    # Acquisition optimization
    n_candidates: int = 2048  # random candidate pool size
    n_starts: int = 25  # L-BFGS-B multistarts from best candidates
    suggestions_per_step: int = 1  # q (number of suggestions per BO iteration)
    diversity_frac: float = 0.05  # fraction of average box-size for min separation
    min_separation: Optional[float] = (
        None  # absolute min separation; overrides diversity_frac
    )

    # GP training
    random_state: Optional[int] = None
    normalize_y: bool = True
    n_restarts_optimizer: int = 5  # for GP hyperparameters
    gp_kwargs: Dict = field(default_factory=dict)

    # runtime fields
    X_: np.ndarray = field(init=False, repr=False)
    y_: np.ndarray = field(init=False, repr=False)
    gp_: GaussianProcessRegressor = field(init=False, repr=False)
    rng_: np.random.Generator = field(init=False, repr=False)
    dim: int = field(init=False, repr=False)

    # debug fields
    acq_opt_callback: Optional[
        Callable[[Any, np.ndarray, np.ndarray, np.ndarray], None]
    ] = None

    def __post_init__(self):
        self.bounds = np.asarray(self.bounds, dtype=float)
        assert (
            self.bounds.ndim == 2 and self.bounds.shape[1] == 2
        ), "bounds must be (d, 2)"
        self.dim = self.bounds.shape[0]
        self.rng_ = np.random.default_rng(self.random_state)
        self.kernel = self.kernel.lower()
        assert self.kernel in ("matern", "rbf"), "kernel must be 'matern' or 'rbf'"
        if self.length_scale_bounds is None:
            self.length_scale_bounds = (1e-3, 1e3)
        if self.c_factor_bounds is None:
            if self.normalize_y:
                self.c_factor_bounds = (1 / 2, 2)
            else:
                self.c_factor_bounds = (1e-3, 1e3)
        # storage
        self.X_ = np.empty((0, self.dim), dtype=float)
        self.y_ = np.empty((0,), dtype=float)
        # init
        self._initialize_design(self.n_init)

    # --------------- Initialization ---------------
    def _uniform_in_bounds(self, n: int) -> np.ndarray:
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        return self.rng_.random((n, self.dim)) * (high - low) + low

    def _initialize_design(self, n: int):
        if self.X_init is not None:
            X0 = self.X_init
        else:
            X0 = self._uniform_in_bounds(n)
        if self.y_init is not None:
            assert (
                self.X_init is not None
            ), "Inital values (y_init) provided without data (X_init)"
            assert len(self.X_init) == len(
                self.y_init
            ), "The number of the initial data points must be the same as the number of corresponding output values"
            y0 = self.y_init
        else:
            if self.f_batch is not None:
                y0 = np.asarray(self.f_batch(X0), dtype=float).reshape(-1)
                assert (
                    y0.shape[0] == X0.shape[0]
                ), "f_batch must return n values for n inputs"
            else:
                y0 = np.array([self._eval(self.func, x) for x in X0], dtype=float)
        self.X_ = np.vstack([self.X_, X0])
        self.y_ = np.concatenate([self.y_, y0])

    def _eval(self, f: Callable[[np.ndarray], float], x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).reshape(-1)
        return float(f(x))

    # --------------- GP Fit ---------------
    def _current_alpha(self) -> Optional[np.ndarray]:
        """Return alpha (noise variance on diagonal) for current dataset, or None."""
        if self.sigma is None:
            return None
        if np.isscalar(self.sigma):
            return float(self.sigma) ** 2
        sig = np.asarray(self.sigma, dtype=float).reshape(-1)
        if sig.size == 1:
            return float(sig[0]) ** 2
        assert (
            sig.size == self.y_.size
        ), "If sigma is array-like, it must match number of observations."
        return sig**2

    def _make_kernel(self):
        # Base kernel with ARD length scales (vector of ones)
        if self.kernel == "matern":
            if self.kernel_isotropic:
                base = Matern(
                    length_scale=1.0,
                    length_scale_bounds=self.length_scale_bounds,
                    nu=2.5,
                )
            else:
                base = Matern(
                    length_scale=np.ones(self.dim),
                    length_scale_bounds=self.length_scale_bounds,
                    nu=2.5,
                )
        else:  # "rbf"
            if self.kernel_isotropic:
                base = RBF(
                    length_scale=1.0,
                    length_scale_bounds=self.length_scale_bounds,
                )
            else:
                base = RBF(
                    length_scale=np.ones(self.dim),
                    length_scale_bounds=self.length_scale_bounds,
                )
        return C(1.0, self.c_factor_bounds) * base

    def _fit_gp(self):
        base_kernel = self._make_kernel()
        if self.sigma is None:
            if self.noise_std_guess is None or self.noise_std_bounds is None:
                raise ValueError(
                    "Provide the observation noise std and its bounds for automatic fitting"
                )
            kernel = base_kernel + WhiteKernel(
                noise_level=self.noise_std_guess**2,
                noise_level_bounds=(
                    self.noise_std_bounds[0] ** 2,
                    self.noise_std_bounds[1] ** 2,
                ),
            )
            alpha = 0.0
        else:
            kernel = base_kernel
            alpha = self._current_alpha()

        self.gp_ = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.rng_.integers(0, 2**31 - 1),
            **self.gp_kwargs,
        )
        self.gp_.fit(self.X_, self.y_)

    # --------------- EI and gradient ---------------
    def _ei(self, X: ArrayLike) -> np.ndarray:
        X = _to_2d(np.asarray(X, dtype=float))
        mu, sigma = self.gp_.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-12)

        y_best = np.min(self.y_) if self.minimize else np.max(self.y_)
        improv = (y_best - mu - self.xi) if self.minimize else (mu - y_best - self.xi)

        Z = improv / sigma
        ei = improv * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei = np.where(sigma <= 1e-12, 0.0, ei)
        return ei

    def _decompose_constant_stationary(self):
        """Extract (amp, length_scales, base_type, base_kernel_object)."""
        K = self.gp_.kernel_
        # Unwrap Sum(Product(C, Base), White) or Product(C, Base)
        prod = None
        if hasattr(K, "k1") and hasattr(K, "k2"):
            # Try to find a Product(C, Base) component
            if (
                hasattr(K.k1, "k1")
                and hasattr(K.k1, "k2")
                and isinstance(getattr(K.k1, "k2", None), (RBF, Matern))
            ):
                prod = K.k1
            elif (
                hasattr(K.k2, "k1")
                and hasattr(K.k2, "k2")
                and isinstance(getattr(K.k2, "k2", None), (RBF, Matern))
            ):
                prod = K.k2
        if prod is None:
            prod = K  # assume already Product(C, Base)

        const = getattr(prod, "k1", None)
        base = getattr(prod, "k2", None)
        amp = float(const.constant_value) if const is not None else 1.0

        if isinstance(base, (RBF, Matern)):
            ell = np.asarray(base.length_scale, dtype=float).reshape(-1)
            if ell.size == 1:
                ell = np.full(self.dim, ell.item())
            base_type = "rbf" if isinstance(base, RBF) else "matern"
        else:
            raise ValueError("Kernel must be Constant * (RBF or Matern).")

        return amp, ell, base_type, base

    def _ei_with_grad(self, x: np.ndarray):
        """Return EI(x) and gradient dEI/dx estimated using central differences."""
        eps = 1e-6
        g_fd = np.zeros_like(x)
        for i in range(x.size):
            e = np.zeros_like(x)
            e[i] = eps
            ei_p = self._ei(x + e)
            ei_m = self._ei(x - e)
            g_fd[i] = (ei_p - ei_m) / (2 * eps)
        return self._ei(x), g_fd

    # --------------- Acquisition maximization (multi) ---------------
    def _nms_select(
        self, X: np.ndarray, scores: np.ndarray, k: int, min_sep: float
    ) -> np.ndarray:
        """Greedy non-maximum suppression: pick up to k points with pairwise distance >= min_sep."""
        order = np.argsort(-scores)  # descending by EI
        chosen = []
        for idx in order:
            x = X[idx]
            if all(np.linalg.norm(x - X[j]) >= min_sep for j in chosen):
                chosen.append(idx)
                if len(chosen) == k:
                    break
        return np.array(chosen, dtype=int)

    def _argmax_acq_multi(self, k: int) -> np.ndarray:
        """Return up to k diverse local maximizers of EI (shape (m<=k, d))."""
        # Random candidate pool
        cand = self._uniform_in_bounds(self.n_candidates)
        acq_vals = self._ei(cand)

        # Best K starts for local refinement
        K = min(self.n_starts, self.n_candidates)
        starts_idx = np.argsort(-acq_vals)[:K]
        starts = cand[starts_idx]

        # Optimize from each start
        xs = []
        vals = []

        def fun(x):
            ei = self._ei(x)
            return -ei  # minimize negative EI

        for x0 in starts:
            res = minimize(
                fun, x0=x0, method="L-BFGS-B", jac=False, bounds=self.bounds.tolist()
            )
            if res.success and np.isfinite(res.fun):
                xs.append(res.x)
                vals.append(-res.fun)  # EI value
        if len(xs) == 0:
            # fallback to random candidates
            xs = [cand[np.argmax(acq_vals)]]
            vals = [float(np.max(acq_vals))]

        xs = np.asarray(xs, dtype=float)
        vals = np.asarray(vals, dtype=float)

        # Merge with top raw candidates to increase diversity pool
        pool_X = np.vstack([xs, starts])
        pool_scores = np.concatenate([vals, acq_vals[starts_idx]])

        # Compute min separation
        if self.min_separation is not None:
            min_sep = float(self.min_separation)
        else:
            ranges = self.bounds[:, 1] - self.bounds[:, 0]
            avg_range = float(np.mean(ranges))
            min_sep = max(self.diversity_frac * avg_range, 1e-12)

        # Greedy NMS selection
        sel_idx = self._nms_select(pool_X, pool_scores, k, min_sep)
        if not self.acq_opt_callback is None:
            self.acq_opt_callback(self, xs, cand, pool_X[sel_idx])
        return pool_X[sel_idx]

    # --------------- ARD helpers ---------------
    def get_length_scales(self) -> np.ndarray:
        """Return a copy of the current per-dimension length scales (ℓ_i)."""
        _, ell, _, _ = self._decompose_constant_stationary()
        return ell.copy()

    def report_ard(
        self,
        feature_names: Optional[List[str]] = None,
        sort_by: str = "inv_length",  # "inv_length" | "length" | "none"
        print_table: bool = True,
        return_data: bool = False,
        precision: int = 6,
    ):
        """Print and/or return ARD report: per-dimension ℓ_i and 1/ℓ_i."""
        ell = self.get_length_scales()
        inv = 1.0 / np.maximum(ell, 1e-12)
        n = ell.size
        names = (
            feature_names if feature_names is not None else [f"x{i}" for i in range(n)]
        )

        if sort_by == "inv_length":
            order = np.argsort(-inv)
        elif sort_by == "length":
            order = np.argsort(ell)
        else:
            order = np.arange(n)

        rows = [
            {
                "feature": names[i],
                "length_scale": float(ell[i]),
                "inv_length": float(inv[i]),
            }
            for i in order
        ]

        if print_table:
            w0 = max(len("feature"), max(len(str(r["feature"])) for r in rows))
            fmt = f"{{:<{w0}s}}  {{:>{precision+8}.{precision}g}}  {{:>{precision+8}.{precision}g}}"
            print(
                f'{"feature":<{w0}s}  {"length_scale":>{precision+8}s}  {"1/length":>{precision+8}s}'
            )
            print("-" * (w0 + 2 + (precision + 8) + 2 + (precision + 8)))
            for r in rows:
                print(fmt.format(r["feature"], r["length_scale"], r["inv_length"]))

        if return_data:
            return rows

    # --------------- Public API ---------------
    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        """One BO iteration. Returns (X_new, y_new).

        If suggestions_per_step == 1:
            - returns X_new shape (1,d), y_new shape (1,)
        Else:
            - returns batched shapes (q,d), (q,)
        """

        # Refit GP
        self._fit_gp()

        # Optimize acq function
        q = int(max(1, self.suggestions_per_step))
        X_new = self._argmax_acq_multi(q)
        print(f"Selected {len(X_new)} new points")

        # Evaluate
        if self.f_batch is not None:
            y_new = np.asarray(self.f_batch(X_new), dtype=float).reshape(-1)
            assert (
                y_new.shape[0] == X_new.shape[0]
            ), "f_batch must return q values for q inputs"
        else:
            y_new = np.array([self._eval(self.func, x) for x in X_new], dtype=float)

        # Update data
        self.X_ = np.vstack([self.X_, X_new])
        self.y_ = np.concatenate([self.y_, y_new])

        return X_new, y_new

    def run(self) -> Dict[str, np.ndarray]:
        for _ in range(self.n_iter):
            self.step()
        i_obs = int(np.argmin(self.y_) if self.minimize else np.argmax(self.y_))
        x_obs, y_obs = self.X_[i_obs].copy(), float(self.y_[i_obs])
        x_mean_in_data, mu_in_data = self.recommend("mean_in_data")
        x_mean_global, mu_global = self.recommend(
            "mean_global", n_starts=max(25, 10 * self.dim)
        )

        return {
            "x_obs_best": x_obs,
            "y_obs_best": y_obs,
            "x_rec_mean_in_data": x_mean_in_data,
            "y_rec_mean_in_data": mu_in_data,
            "x_rec_mean_global": x_mean_global,
            "y_rec_mean_global": mu_global,
            "X": self.X_.copy(),
            "y": self.y_.copy(),
        }

    def get_gp(self) -> GaussianProcessRegressor:
        return self.gp_

    def posterior(self, X: ArrayLike, return_std: bool = True):
        X = _to_2d(np.asarray(X, dtype=float))
        mu, std = self.gp_.predict(X, return_std=True)
        if return_std:
            return mu, std
        return mu, None

    def recommend(self, mode="mean_global", n_starts=25):
        if mode == "observed":
            i = int(np.argmin(self.y_) if self.minimize else np.argmax(self.y_))
            return self.X_[i].copy(), float(self.y_[i])

        self._fit_gp() # Add fitting in case it is not done after the data last update
        mu, _ = self.gp_.predict(self.X_, return_std=True)
        if mode == "mean_in_data":
            i = int(np.argmin(mu) if self.minimize else np.argmax(mu))
            return self.X_[i].copy(), float(mu[i])

        def fun(x):
            m, _ = self.gp_.predict(np.asarray(x).reshape(1, -1), return_std=True)
            return float(m[0])

        starts = self._uniform_in_bounds(n_starts)
        best_x, best_v = None, np.inf
        for x0 in starts:
            res = minimize(fun, x0=x0, method="L-BFGS-B", bounds=self.bounds.tolist())
            if res.success and res.fun < best_v:
                best_x, best_v = res.x, res.fun
        return best_x, float(best_v)

    # --------------- Minimal portable snapshot ---------------
    def save_snapshot(self, save_dir: str, results: Dict = None) -> Path:
        """
        Сохраняет минимальный снимок:
          - gp.joblib        : GaussianProcessRegressor (joblib.dump)
          - arrays.npz       : все np.ndarray из атрибутов объекта
          - manifest.json    : дата + версии numpy/scikit-learn
          - result.json      : x_best, y_best, формы X/y
        """
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)

        # 1) GP → joblib
        joblib.dump(self.gp_, d / "gp.joblib", compress=3)

        # 2) Соберём все NumPy-массивы из атрибутов
        arrays = {}
        for k, v in self.__dict__.items():
            if k == "gp_":
                continue
            if isinstance(v, np.ndarray):
                arrays[k] = v

        # На всякий случай убедимся, что ключевые массивы попали
        if "X_" not in arrays and hasattr(self, "X_"):
            arrays["X_"] = np.asarray(self.X_)
        if "y_" not in arrays and hasattr(self, "y_"):
            arrays["y_"] = np.asarray(self.y_)
        if (
            "bounds" not in arrays
            and hasattr(self, "bounds")
            and isinstance(self.bounds, np.ndarray)
        ):
            arrays["bounds"] = np.asarray(self.bounds)

        np.savez_compressed(d / "arrays.npz", **arrays)

        # 3) manifest (дата + версии)
        manifest = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
        }
        (d / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        # 4) result (best point) + формы
        X = arrays.get("X_", np.empty((0, getattr(self, "dim", 1))))
        y = arrays.get("y_", np.empty((0,), dtype=float))
        if results is None:
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
                "X_shape": list(X.shape),
                "y_shape": list(y.shape),
            }

        (d / "result.json").write_text(
            json.dumps(result_json, indent=2), encoding="utf-8"
        )
        return d

    @staticmethod
    def load_snapshot(load_dir: str) -> "BOSnapshot":
        """
        Загружает минимальный снимок, сохранённый save_snapshot(...).
        Возвращает BOSnapshot (gp, arrays, manifest, result).
        """
        d = Path(load_dir)
        gp = joblib.load(d / "gp.joblib")
        arrays_file = np.load(d / "arrays.npz")
        arrays = {k: arrays_file[k] for k in arrays_file.files}
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        result = json.loads((d / "result.json").read_text(encoding="utf-8"))
        return BOSnapshot(gp=gp, arrays=arrays, manifest=manifest, result=result)
