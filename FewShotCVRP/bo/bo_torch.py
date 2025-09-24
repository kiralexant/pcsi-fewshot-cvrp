from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.acquisition.analytic import ExpectedImprovement, PosteriorMean
from botorch.exceptions.warnings import NumericsWarning  # BoTorch EI numerics
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior, UniformPrior
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.stats import norm, qmc
from FewShotCVRP.utils.logs import configure_logger, timecall
from .gp_fitting import MLLFitConfig, optimize_mll


def _to_2d_np64(X: ArrayLike) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X


def _as_t(
    X: ArrayLike, dtype: torch.dtype, device: Union[str, torch.device]
) -> torch.Tensor:
    return torch.as_tensor(np.asarray(X, dtype=np.float64), dtype=dtype, device=device)


def _to_numpy64(x_t: torch.Tensor) -> np.ndarray:
    return x_t.detach().cpu().numpy().astype(np.float64, copy=False)


def _has_scalekernel(gp) -> bool:
    from gpytorch.kernels import ScaleKernel

    return isinstance(gp.covar_module, ScaleKernel)


def _base_kernel(gp):
    return gp.covar_module.base_kernel if _has_scalekernel(gp) else gp.covar_module


@dataclass
class BOSnapshot:
    gp: Any  # BoTorch model
    arrays: Dict[str, np.ndarray]  # all np-atributes (X_, y_, bounds, …)
    manifest: Dict[str, Any]  # versions/date
    result: Dict[str, Any]  # x_best / y_best / shapes
    experiment_config: Dict[str, Any]


@dataclass
class BayesianOptimizer:
    # Objective(s)
    func: Callable[[np.ndarray], float]
    f_batch: Optional[Callable[[np.ndarray], ArrayLike]] = None

    # Initial design (optional)
    X_init: Optional[ArrayLike] = None
    y_init: Optional[ArrayLike] = None

    # Domain & BO config
    bounds: Sequence[Tuple[float, float]] = ()
    n_init: int = 5
    n_iter: int = 25
    minimize: bool = True

    # Noise (None => infer; else fixed)
    sigma: Optional[Union[float, np.ndarray]] = None  # observation std

    # Kernel / model config
    kernel: str = "matern"  # "matern" | "rbf"
    matern_nu: float = (2.5,)
    kernel_isotropic: bool = False
    use_input_normalize: bool = True
    normalize_y: bool = True
    add_vanilla_bo_prior: bool = True
    mll_fit_config: MLLFitConfig = field(default_factory=MLLFitConfig)

    # Acquisition optimization
    acq_function: str = "logEI"  # "EI" | "logEI"
    n_init_samples_acq_opt: int = 2048  # raw_samples
    n_restarts_acq_opt: int = 20  # num_restarts
    suggestions_per_step: int = 10  # q (generated sequentially)
    diversity_frac: float = 0.1  # fraction of average box-size for min separation
    min_separation: Optional[float] = (
        None  # absolute min separation; overrides diversity_frac
    )

    # Randomness & extras
    random_state: Optional[int] = None
    gp_kwargs: Dict = field(default_factory=dict)

    # MC settings (for potential sampling use; we don't use qEI now)
    mc_samples: int = 128

    # DoE
    doe_method: str = "sobol"  # "precomputed" | "sobol" | "random" | "lhs"
    doe_seed: Optional[int] = None  # (deprecated, use random_state)

    # Device / dtype
    device: Union[str, torch.device] = "cpu"
    dtype: torch.dtype = torch.double

    # runtime
    X_: torch.Tensor = field(init=False, repr=False)  # (n,d) device/dtype
    y_: torch.Tensor = field(init=False, repr=False)  # (n,1) device/dtype
    gp_: Any = field(init=False, repr=False)
    rng_: np.random.Generator = field(init=False, repr=False)
    dim: int = field(init=False, repr=False)

    # debug
    logger: Optional[logging.Logger] = None
    acq_opt_callback: Optional[
        Callable[[Any, np.ndarray, np.ndarray, np.ndarray], None]
    ] = None

    def __post_init__(self):
        # Cache device
        self.device = torch.device(self.device)
        self.bounds_np = np.asarray(self.bounds, dtype=np.float64)
        lb_t = torch.as_tensor(
            self.bounds_np[:, 0], dtype=self.dtype, device=self.device
        )
        ub_t = torch.as_tensor(
            self.bounds_np[:, 1], dtype=self.dtype, device=self.device
        )
        self.bounds_t = torch.stack([lb_t, ub_t])  # [2, d]
        assert (
            self.bounds_np.ndim == 2 and self.bounds_np.shape[1] == 2
        ), "bounds must be (d, 2)"
        self.dim = self.bounds_np.shape[0]
        self.kernel = self.kernel.lower()
        assert self.kernel in ("matern", "rbf"), "kernel must be 'matern' or 'rbf'"
        assert self.acq_function in (
            "EI",
            "logEI",
        ), "acq_function must be 'EI' or 'logEI'"

        # Seed all RNGs consistently
        self.rng_ = np.random.default_rng(self.random_state)
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.random_state))

        # Keep bounds both as numpy (for SciPy) and as a tensor on device for BoTorch
        self.bounds_np = np.asarray(self.bounds, dtype=np.float64)
        lb_t = torch.as_tensor(
            self.bounds_np[:, 0], dtype=self.dtype, device=self.device
        )
        ub_t = torch.as_tensor(
            self.bounds_np[:, 1], dtype=self.dtype, device=self.device
        )
        self.bounds_t = torch.stack([lb_t, ub_t])  # [2, d]

        if self.logger is None:
            self.logger = configure_logger("FewShotCVRPLogger")
        self._fit_gp = timecall(self.logger, "fit_gp")(self._fit_gp)
        self._argmax_acq_multi = timecall(self.logger, "argmax_acq")(
            self._argmax_acq_multi
        )
        self._evaluate_objective_t = timecall(self.logger, "evaluate_objective")(
            self._evaluate_objective_t
        )
        self.step = timecall(self.logger, "BO step")(self.step)

    # ---------------- Initialization ----------------
    def _sample_doe(self, n: int) -> np.ndarray:
        """Return n samples in the box (numpy float64)."""
        d = self.dim
        m = self.doe_method.lower()
        if m == "precomputed":
            return self.X_init[:n]
        if m == "sobol":
            bounds_t = self.bounds_t
            X = draw_sobol_samples(
                bounds=bounds_t, n=n, q=1, seed=self.random_state
            ).squeeze(-2)
            return _to_numpy64(X)  # (n,d)
        if m == "random":
            lb = self.bounds_np[:, 0]
            ub = self.bounds_np[:, 1]
            U = self.rng_.random((n, d), dtype=np.float64)
            return lb + (ub - lb) * U
        if m == "lhs":
            sampler = qmc.LatinHypercube(d=d, seed=self.random_state)
            U = sampler.random(n=n)
            return qmc.scale(U, self.bounds_np[:, 0], self.bounds_np[:, 1]).astype(
                np.float64
            )
        raise ValueError(f"Unknown doe_method={self.doe_method!r}")

    def _initialize_design(self):
        # Build initial design on correct device as tensors
        if self.X_init is not None:
            X0_np = _to_2d_np64(self.X_init)
            if self.n_init is None:
                self.n_init = len(X0_np)
        X0_np = self._sample_doe(self.n_init)  # returns np.float64
        if self.y_init is not None:
            y0_np = np.asarray(self.y_init, dtype=np.float64).reshape(-1)[: self.n_init]
            assert X0_np.shape[0] == y0_np.shape[0], "X_init and y_init size mismatch"
        else:
            # Evaluate with the unified evaluator (handles f or f_batch)
            y0_np = self._evaluate_objective_np(X0_np)

        # Store as tensors (BoTorch-facing)
        self.X_ = _as_t(X0_np, self.dtype, self.device)  # (n,d)
        self.y_ = _as_t(y0_np.reshape(-1, 1), self.dtype, self.device)  # (n,1)

    # ---------------- objective evaluation ----------------
    def _evaluate_objective_np(self, X_np: np.ndarray) -> np.ndarray:
        """Evaluate objective(s) given numpy float64 array (n,d) -> returns numpy (n,)."""
        if self.f_batch is not None:
            y = np.asarray(
                self.f_batch(np.asarray(X_np, dtype=np.float64)), dtype=np.float64
            ).reshape(-1)
            assert (
                y.shape[0] == X_np.shape[0]
            ), "f_batch must return n values for n inputs"
            return y
        # scalar objective
        ys = [
            float(self.func(np.asarray(x, dtype=np.float64).reshape(-1))) for x in X_np
        ]
        return np.asarray(ys, dtype=np.float64)

    def _evaluate_objective_t(self, X_t: torch.Tensor) -> torch.Tensor:
        """Evaluate objective(s) for a tensor (q,d) on device -> tensor (q,1) on device."""
        X_np = _to_numpy64(X_t)  # (q,d)
        y_np = self._evaluate_objective_np(X_np)  # (q,)
        return _as_t(y_np.reshape(-1, 1), self.dtype, self.device)

    # ---------------- GP fit ----------------
    def _build_covar_module(self):
        ard = None if self.kernel_isotropic else self.dim

        lengthscale_prior = None
        if self.add_vanilla_bo_prior:
            mu = np.sqrt(2.0) + 0.5 * np.log(self.dim)
            sigma = np.sqrt(3.0)
            lengthscale_prior = LogNormalPrior(loc=mu, scale=sigma)
        # lengthscale_prior = UniformPrior(a=1.0, b=10**4)

        base = (
            MaternKernel(
                nu=self.matern_nu, ard_num_dims=ard, lengthscale_prior=lengthscale_prior
            )
            if self.kernel == "matern"
            else RBFKernel(ard_num_dims=ard, lengthscale_prior=lengthscale_prior)
        )
        new_ls = _as_t(
            np.random.uniform(10.0, 10**3, self.dim).reshape(1, -1),
            device=self.device,
            dtype=self.dtype,
        )
        base.lengthscale = new_ls
        return base

    def _create_gp_model(self, X: torch.Tensor, Y: torch.Tensor):
        input_tf = Normalize(self.dim) if self.use_input_normalize else None
        outcome_tf = Standardize(m=1) if self.normalize_y else None
        covar_module = self._build_covar_module()

        # Fixed or inferred noise via SingleTaskGP
        if self.sigma is not None:
            if np.isscalar(self.sigma):
                yvar_np = np.full(
                    (Y.shape[0], 1), float(self.sigma) ** 2, dtype=np.float64
                )
            else:
                sig = np.asarray(self.sigma, dtype=np.float64).reshape(-1)
                assert sig.size in (1, int(Y.shape[0]))
                sig_np = (
                    sig
                    if sig.size > 1
                    else np.full(int(Y.shape[0]), sig[0], dtype=np.float64)
                )
                yvar_np = (sig_np**2).reshape(-1, 1)
            Yvar = _as_t(yvar_np, self.dtype, self.device)
            model = SingleTaskGP(
                train_X=X,
                train_Y=Y,
                train_Yvar=Yvar,
                input_transform=input_tf,
                outcome_transform=outcome_tf,
                covar_module=covar_module,
                **self.gp_kwargs,
            )
        else:
            model = SingleTaskGP(
                train_X=X,
                train_Y=Y,
                input_transform=input_tf,
                outcome_transform=outcome_tf,
                covar_module=covar_module,
                **self.gp_kwargs,
            )
        return model

    def _fit_gp(self):
        self.logger.info(f"Fitting GP with X shape: {self.X_.shape}, y shape: {self.y_.shape} ")
        gp_model = self._create_gp_model(self.X_, self.y_)
        gp_model, info = optimize_mll(
            gp_model, seed=self.rng_.integers(0, 10**9), cfg=self.mll_fit_config
        )
        self.gp_ = gp_model.to(self.device, dtype=self.dtype)

    # ---------------- Optimize acquisition function ----------------
    def _nms_select(
        self, X: torch.Tensor, scores: torch.Tensor, k: int, min_sep: float
    ) -> torch.Tensor:
        scores = scores.view(-1)  # ensure (n,)
        order = torch.argsort(scores, descending=True)
        n = X.shape[0]
        keep: list[int] = []
        suppressed = torch.zeros(n, dtype=torch.bool, device=self.device)

        min_sep_sq = float(min_sep) ** 2
        for idx in order:
            if suppressed[idx]:
                continue
            keep.append(int(idx))
            if len(keep) == k:
                break
            # suppress all points within min_sep of the newly chosen point (use squared distances)
            diff = X - X[idx]  # (n, d)
            d2 = (diff * diff).sum(dim=1)  # (n,)
            suppressed |= d2 < min_sep_sq

        return torch.tensor(keep, dtype=torch.int, device=self.device)

    def _build_ei_acqf(self):
        # best_f on transformed scale
        if self.minimize:
            best_f = (self.y_).min().detach()
        else:
            best_f = self.y_.max().detach()
        if self.acq_function == "EI":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NumericsWarning)
                acqf = ExpectedImprovement(
                    model=self.gp_, best_f=best_f, maximize=(not self.minimize)
                )
        elif self.acq_function == "logEI":
            acqf = LogExpectedImprovement(
                model=self.gp_, best_f=best_f, maximize=(not self.minimize)
            )
        return acqf

    def __log_selected_score(self, pool_scores, sel_idx, acqf):
        with torch.no_grad():
            sel_scores_t = pool_scores[sel_idx].view(-1)
            sel_scores = sel_scores_t.detach().cpu().numpy().tolist()
            # One reference "starting" score (random Sobol point within bounds)
            baseline_X = draw_sobol_samples(self.bounds_t, n=1, q=1).squeeze(1)
            baseline_score = acqf(baseline_X).detach().view(-1)[0].item()
        # Pretty log
        if hasattr(self, "logger") and self.logger is not None:
            top_k = min(len(sel_scores), 8)
            top_vals = ", ".join(f"{v:.6g}" for v in sel_scores[:top_k])
            more = (
                f" (+{len(sel_scores)-top_k} more)" if len(sel_scores) > top_k else ""
            )
            self.logger.info(
                "EI scores (selected): [%s]%s | baseline(start) ≈ %.6g",
                top_vals,
                more,
                baseline_score,
            )

    def _argmax_acq_multi(self, k: int) -> torch.Tensor:
        # Base EI built once
        acqf = self._build_ei_acqf()

        # Optimize from each start
        num_restarts = max(1, int(self.n_restarts_acq_opt))
        raw_samples = max(256, int(self.n_init_samples_acq_opt))
        pool_X, pool_scores = optimize_acqf(
            acqf,
            bounds=self.bounds_t,
            q=1,
            return_best_only=False,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            sequential=True,
        )
        pool_X = pool_X.squeeze(1)

        # Compute min separation
        if self.min_separation is not None:
            min_sep = float(self.min_separation)
        else:
            ranges = self.bounds_t[1] - self.bounds_t[0]
            avg_range = torch.mean(ranges)
            min_sep = max(self.diversity_frac * avg_range, 1e-12)

        # Greedy NMS selection
        k = max(1, int(k))
        sel_idx = self._nms_select(pool_X, pool_scores, k, min_sep)
        if not self.acq_opt_callback is None:
            self.acq_opt_callback(
                self, _to_numpy64(pool_X), None, _to_numpy64(pool_X[sel_idx])
            )
        self.__log_selected_score(pool_scores, sel_idx, acqf)
        return pool_X[sel_idx], acqf

    # ---------------- Public API ----------------
    def posterior(self, X: torch.Tensor, return_std: bool = True):
        with torch.no_grad():
            post = self.gp_.posterior(X)
            mu_t = post.mean
            var_t = post.variance if return_std else None
        mu = _to_numpy64(mu_t.squeeze(-1))
        if return_std:
            std = _to_numpy64(torch.sqrt(var_t).squeeze(-1))
            return mu, std
        return mu, None

    def step(self) -> None:
        self._fit_gp()
        X_new_t, _ = self._argmax_acq_multi(self.suggestions_per_step)  # (q,d)
        y_new_t = self._evaluate_objective_t(X_new_t)  # (q,1)

        # Update training tensors
        self.X_ = torch.cat([self.X_, X_new_t.to(self.device, dtype=self.dtype)], dim=0)
        self.y_ = torch.cat([self.y_, y_new_t.to(self.device, dtype=self.dtype)], dim=0)

    def run(self) -> Dict[str, np.ndarray]:
        self._initialize_design()
        for self.current_iteration in range(self.n_iter):
            self.step()
            self.logger.info(
                f"Step {self.current_iteration+1} finished. BO progress: {self.current_iteration+1}/{self.n_iter}"
            )

        self._fit_gp()
        x_obs, y_obs = self.recommend("observed", refit_gp=False)
        x_mean_in_data, y_mean_in_data = self.recommend("mean_in_data", refit_gp=False)
        x_rec_mean_global, y_rec_mean_global = self.recommend(
            "mean_global", refit_gp=False
        )

        return {
            "x_obs_best": x_obs,
            "y_obs_best": y_obs,
            "x_rec_mean_in_data": x_mean_in_data,
            "y_rec_mean_in_data": y_mean_in_data,
            "x_rec_mean_global": x_rec_mean_global,
            "y_rec_mean_global": y_rec_mean_global,
            "X": _to_numpy64(self.X_),
            "y": _to_numpy64(self.y_.squeeze(-1)),
        }

    def get_gp(self) -> Any:
        return self.gp_

    def recommend(
        self, mode="mean_global", refit_gp=True, n_restarts_opt_mean_global=100
    ):
        if refit_gp:
            self._fit_gp()
        if mode == "observed":
            y_flat = self.y_.squeeze(-1)
            i = int(
                torch.argmin(y_flat).item()
                if self.minimize
                else torch.argmax(y_flat).item()
            )
            return _to_numpy64(self.X_[i]), float(_to_numpy64(y_flat[i]))

        mu, _ = self.posterior(self.X_, return_std=True)
        if mode == "mean_in_data":
            i = int(np.argmin(mu) if self.minimize else np.argmax(mu))
            return _to_numpy64(self.X_[i]), float(mu[i])

        acq = PosteriorMean(model=self.gp_, maximize=False)  # minimize the mean
        X_star, y_star = optimize_acqf(
            acq,
            bounds=self.bounds_t,
            q=1,
            num_restarts=n_restarts_opt_mean_global,
            raw_samples=100 * self.dim,
        )
        x_star = _to_numpy64(X_star.squeeze(0).squeeze(0))
        y_star = _to_numpy64(y_star.squeeze(0))
        return x_star, float(-y_star)

    # ---------------- snapshot ----------------
    def save_snapshot(self, save_dir: str, results: Dict = None) -> Path:
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)

        # Save model params
        if hasattr(self, "gp_"):
            torch.save(self.gp_.state_dict(), d / "gp_state.pt")

        arrays = {
            "X_": _to_numpy64(self.X_),
            "y_": _to_numpy64(self.y_.squeeze(-1)),
            "bounds": np.asarray(self.bounds_np, dtype=np.float64),
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
                "X_shape": list(arrays["X_"].shape),
                "y_shape": list(arrays["y_"].shape),
            }
        (d / "result.json").write_text(
            json.dumps(result_json, indent=2), encoding="utf-8"
        )
        return d

    @staticmethod
    def load_snapshot(load_dir: str) -> "BOSnapshot":
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
        return BOSnapshot(
            gp=gp_state,
            arrays=arrays,
            manifest=manifest,
            result=result,
            experiment_config=experiment_config,
        )

    # ---------------- ARD report ----------------
    def get_length_scales(self, gp: Any) -> np.ndarray:
        ell_t = _base_kernel(gp).lengthscale.detach()  # (1,d) or (1,1)
        ell = _to_numpy64(ell_t).reshape(-1)
        if self.kernel_isotropic and ell.size == 1:
            ell = np.full(self.dim, float(ell[0]))
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
        ell = self.get_length_scales(self.get_gp())
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
