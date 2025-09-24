import logging
import math
from copy import deepcopy
from functools import wraps
from typing import Any, Dict, Optional, Tuple

import cma
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.optim.core import OptimizationStatus
from gpytorch.kernels import ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from pydantic import BaseModel, ConfigDict


def sample_lengthscales_torch(
    d: int,
    n: int = 1,
    mu: float = 0.0,
    sigma: float = 1.0,
    device=None,
    dtype=torch.double,
) -> torch.Tensor:
    """
    Returns ARD lengthscales with shape:
      - (1, 1, d) if n == 1
      - (n, 1, 1, d) if n > 1  (for multi-restart inits)
    """
    device = device or torch.device("cpu")
    dist = torch.distributions.LogNormal(
        loc=torch.tensor(mu, dtype=dtype, device=device),
        scale=torch.tensor(sigma, dtype=dtype, device=device),
    )
    if n == 1:
        ls = dist.sample((d,)).view(1, 1, d)
    else:
        ls = dist.sample((n, d)).view(n, 1, 1, d)
    return ls


def sample_lengthscales_vanilla_bo(d: int, n: int = 1, device=None, dtype=torch.double):
    mu = math.sqrt(2.0) + 0.5 * math.log(d)  # dimension-scaled
    sigma = math.sqrt(3.0)
    return sample_lengthscales_torch(
        d=d, n=n, mu=mu, sigma=sigma, device=device, dtype=dtype
    )


def sample_lengthscales_uniform(
    d: int,
    n: int = 1,
    low: float = 1.0,
    high: float = 10.0**4,
    device=None,
    dtype=torch.double,
) -> torch.Tensor:
    """
    Uniform sampler on [low, high] (ARD):
      - (1, 1, d) if n == 1
      - (n, 1, 1, d) if n > 1
    """
    device = device or torch.device("cpu")
    lo = torch.as_tensor(low, dtype=dtype, device=device)
    hi = torch.as_tensor(high, dtype=dtype, device=device)
    dist = torch.distributions.Uniform(lo, hi)
    if n == 1:
        ls = dist.sample((d,)).view(1, 1, d)
    else:
        ls = dist.sample((n, d)).view(n, 1, 1, d)
    return ls


# ---------- Utilities ----------


def ensure_train_mode(model_arg=0, include_likelihood=True):
    """
    Decorator factory: ensure a GP model is in .train() during the call,
    then restore previous mode(s).

    Parameters
    ----------
    model_arg : int | str | callable
        How to locate the model in your function's arguments:
          - int  -> positional index in *args (default: 0)
          - str  -> keyword name in **kwargs
          - callable -> resolver (args, kwargs) -> model
    include_likelihood : bool
        If True, also toggle model.likelihood (when present).

    Usage
    -----
    @ensure_train_mode()                 # model is first positional arg
    def step(model, X, Y): ...

    @ensure_train_mode(model_arg="gp")   # model passed as keyword 'gp'
    def step2(*, gp, X, Y): ...

    @ensure_train_mode(model_arg=lambda a, k: k["gp"])  # custom resolver
    def step3(*args, **kwargs): ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # --- resolve the model ---
            if callable(model_arg):
                model = model_arg(args, kwargs)
            elif isinstance(model_arg, int):
                model = args[model_arg]
            elif isinstance(model_arg, str):
                model = kwargs[model_arg]
            else:
                raise TypeError("model_arg must be int, str, or callable")
            if model is None:
                raise ValueError("Could not resolve the GP model")

            # --- capture current modes ---
            prev_model_training = bool(getattr(model, "training", False))
            likelihood = (
                getattr(model, "likelihood", None) if include_likelihood else None
            )
            prev_lik_training = (
                bool(likelihood.training) if likelihood is not None else None
            )

            try:
                # --- ensure train mode for the call ---
                if not prev_model_training:
                    model.train()
                if likelihood is not None and not prev_lik_training:
                    likelihood.train()
                return func(*args, **kwargs)
            finally:
                # --- restore previous modes ---
                model.train(prev_model_training)
                if likelihood is not None:
                    likelihood.train(prev_lik_training)

        return wrapper

    return decorator


@ensure_train_mode()
def compute_likelihood(gp):
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    Xtr = gp.train_inputs[0]
    Ytr = gp.train_targets
    with torch.no_grad():
        return float(torch.sum(mll(gp(Xtr), Ytr)).item())


@ensure_train_mode()
def trainpath_mll_and_grad(
    gp,
    return_numpy: bool = True,
    retain_graph: bool = False,
):
    """
    Compute the exact training-path loss and its gradient.

    Returns
    -------
    loss_value : float or torch.Tensor
        The scalar loss value (exact MLL, i.e., sum(ExactMarginalLogLikelihood)).
    grad_vec : np.ndarray or torch.Tensor, shape (P,)
        Concatenated gradient vector over all MLL parameters that require grad
        (includes model AND likelihood parameters; grads are w.r.t. unconstrained/raw params).
    grad_dict : dict[str, torch.Tensor]
        Mapping param name -> gradient tensor in the parameter's shape.

    Notes
    -----
    - Uses the model's cached training tensors (`gp.train_inputs[0]`, `gp.train_targets`),
      which is the correct target if you use outcome transforms.
    - Gradients are taken for the MLL module's parameters (mll.parameters()),
      which includes both model and likelihood parameters as used by BoTorch fitting.
    """
    # Build MLL once
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    Xtr = gp.train_inputs[0]
    Ytr = gp.train_targets

    # Forward (exact train objective)
    out = gp(Xtr)
    loss = torch.sum(mll(out, Ytr))  # may be batched -> sum to scalar

    # Collect the parameters we differentiate
    params = [(n, p) for (n, p) in mll.named_parameters() if p.requires_grad]
    if not params:
        # nothing to optimize
        loss_value = float(loss.item()) if return_numpy else loss
        empty = (
            (torch.empty(0), {}) if not return_numpy else (torch.empty(0).numpy(), {})
        )
        return (loss_value, *empty)

    names, tensors = zip(*params)

    # Autograd without touching .grad buffers (no accumulation)
    grads = torch.autograd.grad(
        loss,
        tensors,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )

    # Build outputs
    grad_chunks = []
    grad_dict = {}
    for (name, p), g in zip(params, grads):
        if g is None:
            g = torch.zeros_like(p)
        grad_dict[name] = g.detach()
        grad_chunks.append(g.reshape(-1))

    grad_vec_t = torch.cat(grad_chunks).detach()
    if return_numpy:
        return float(loss.item()), grad_vec_t.cpu().numpy(), grad_dict
    else:
        return loss.detach(), grad_vec_t, grad_dict


# --------------------------- small helpers ---------------------------
@torch.no_grad()
def _init_lengthscales_vanilla_bo_(gp, seed: int | None = None) -> bool:
    """
    Initialize kernel lengthscales using the Vanilla-BO sampler before optimization.
    Returns True if lengthscales were initialized, False if there is nothing to do.
    """
    base = _base_kernel(gp)
    # only if lengthscales exist and are optimizable
    if not (hasattr(base, "raw_lengthscale") and base.raw_lengthscale.requires_grad):
        return False

    # optional reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    d = base.lengthscale.numel()  # ARD dims or 1 if not ARD
    device = base.lengthscale.device
    dtype = base.lengthscale.dtype

    ls0 = sample_lengthscales_vanilla_bo(
        d=d, n=1, device=device, dtype=dtype
    )  # (1,1,d)
    base.lengthscale = ls0  # already the correct ARD-safe shape
    return True


def _has_scalekernel(gp) -> bool:
    return isinstance(gp.covar_module, ScaleKernel)


def _base_kernel(gp):
    return gp.covar_module.base_kernel if _has_scalekernel(gp) else gp.covar_module


def _trainpath_mll(gp) -> float:
    """
    Exact train-path MLL on cached training tensors.
    Assumes that gp is in training mode.
    """
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    Xtr = gp.train_inputs[0]
    Ytr = gp.train_targets
    with torch.no_grad():
        return float(torch.sum(mll(gp(Xtr), Ytr)).item())


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1)


def _sigma0_from_bounds(lo: np.ndarray, hi: np.ndarray, rule: str = "quarter") -> float:
    r"""
    Compute a default initial CMA-ES step-size :math:`\sigma_0` from elementwise bounds
    in the **decision space** (the same space as the CMA-ES vector :math:`\mathbf{z}`).
    For background on step-size in CMA-ES (and its adaptation), see Hansen's tutorial. [1]

    Heuristics
    ----------
    - **Quarter-width rule (default)**:
      :math:`\sigma_0 = 0.25 \cdot \mathrm{median}( \mathbf{h} - \mathbf{l} )`.
      Recommended in the official *pycma* documentation:
      “``sigma0`` should be about **1/4th of the search domain width** (where the optimum is to be expected).”
      [2]

    - **Conservative min-sixth rule**:
      :math:`\sigma_0 = \min(\mathbf{h} - \mathbf{l}) / 6`.
      Used by **Optuna**'s CMA-ES samplers as a robust default across heterogeneous search spaces. [3]

    Parameters
    ----------
    lo : np.ndarray
        Lower bounds per coordinate (same shape as the decision vector).
    hi : np.ndarray
        Upper bounds per coordinate (same shape as the decision vector).
    rule : {"quarter", "min_sixth"}, default "quarter"
        Heuristic to use (see above).

    Returns
    -------
    float
        Suggested initial step-size :math:`\sigma_0`.

    References
    ----------
    [1] N. Hansen. “The CMA Evolution Strategy: A Tutorial.” *arXiv* 1604.00772, 2016.
        https://arxiv.org/abs/1604.00772

    [2] N. Hansen, Y. Akimoto, and P. Baudis. *CMA-ES/pycma* documentation,
        “cma.evolution_strategy - CMAEvolutionStrategy”.
        https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.html  (accessed Sep 2025).

    [3] Optuna Developers. *Optuna* documentation,
        “optuna.samplers.CmaEsSampler”.
        https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.CmaEsSampler.html  (accessed Sep 2025).
    """
    widths = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
    widths = np.maximum(widths, 1e-12)  # numerical safety

    if rule == "quarter":
        sigma0 = float(np.median(widths) / 4.0)
    elif rule == "min_sixth":
        sigma0 = float(np.min(widths) / 6.0)
    else:
        raise ValueError(f"Unknown sigma0 rule: {rule}")

    return max(sigma0, 1e-12)


# ----------------------- discover/pack parameters ---------------------


def discover_mll_params(gp):
    """
    Inspect model and return a list of 'items' describing all optimizable
    parameters that influence the exact MLL:
      - base_kernel.lengthscale (ARD/batched ok)  -> positive (log-transform)
      - covar_module.outputscale (if ScaleKernel) -> positive (log)
      - likelihood.noise (if GaussianLikelihood)  -> positive (log)
      - mean_module.constant (if ConstantMean)    -> real (identity)
      - base_kernel.period_length (if present)    -> positive (log)
    Only include if the underlying raw parameter requires_grad.
    """
    logger = logging.getLogger("FewShotCVRPLogger")
    items = []

    # base kernel lengthscale
    base = _base_kernel(gp)
    if hasattr(base, "raw_lengthscale") and base.raw_lengthscale.requires_grad:
        val = base.lengthscale.detach()
        items.append(
            {
                "name": "lengthscale",
                "getter": lambda: _flatten(_base_kernel(gp).lengthscale.detach()),
                "setter": lambda v: setattr(
                    _base_kernel(gp),
                    "lengthscale",
                    v.view(*((1,) * (3 - len(v.shape))), -1),
                ),
                "transform": "logpos",
                # "transform": "identity",
                "shape": _flatten(val).shape,
            }
        )
        logger.info(f"Found lengthscales parameters, shape: { _flatten(val).shape}")

    # outputscale (ScaleKernel)
    if _has_scalekernel(gp) and gp.covar_module.raw_outputscale.requires_grad:
        items.append(
            {
                "name": "outputscale",
                "getter": lambda: gp.covar_module.outputscale.detach().view(-1),
                "setter": lambda v: setattr(
                    gp.covar_module,
                    "outputscale",
                    v.view(()).item() if v.numel() == 1 else v,
                ),
                # "transform": "logpos",
                "transform": "identity",
                "shape": torch.Size([1]),
            }
        )
        logger.info(f"Found output scale parameters, shape: (1)")

    # likelihood noise (GaussianLikelihood)
    if hasattr(gp.likelihood, "noise_covar") and hasattr(
        gp.likelihood.noise_covar, "raw_noise"
    ):
        if gp.likelihood.noise_covar.raw_noise.requires_grad:
            items.append(
                {
                    "name": "noise",
                    "getter": lambda: gp.likelihood.noise.detach().view(-1),
                    "setter": lambda v: setattr(
                        gp.likelihood,
                        "noise",
                        v.view(()).item() if v.numel() == 1 else v,
                    ),
                    "transform": "logpos",
                    # "transform": "identity",
                    "shape": torch.Size([1]),
                }
            )
            logger.info(f"Found output noise parameter, shape: (1)")

    # mean constant (ConstantMean)
    if hasattr(gp, "mean_module") and hasattr(gp.mean_module, "constant"):
        if gp.mean_module.constant.requires_grad:
            items.append(
                {
                    "name": "mean_constant",
                    "getter": lambda: gp.mean_module.constant.detach().view(-1),
                    "setter": lambda v: gp.mean_module.constant.copy_(
                        v.view_as(gp.mean_module.constant)
                    ),
                    "transform": "identity",
                    "shape": gp.mean_module.constant.shape,
                }
            )
            logger.info(
                f"Found constant mean parameter, shape: {gp.mean_module.constant.shape}"
            )

    # period_length (e.g., PeriodicKernel) if present
    if hasattr(base, "raw_period_length"):
        if base.raw_period_length.requires_grad:
            items.append(
                {
                    "name": "period_length",
                    "getter": lambda: _flatten(_base_kernel(gp).period_length.detach()),
                    "setter": lambda v: setattr(
                        _base_kernel(gp),
                        "period_length",
                        v.view(*((1,) * (3 - len(v.shape))), -1),
                    ),
                    # "transform": "logpos",
                    "transform": "identity",
                    "shape": _flatten(base.period_length.detach()).shape,
                }
            )
            logger.info(
                f"Found periodic length parameter in periodic kernel, shape: {_flatten(base.period_length.detach()).shape}"
            )

    return items


def pack_to_z(gp, items):
    """Concatenate transformed params into a 1D numpy vector z."""
    zs = []
    for it in items:
        val = it["getter"]()  # tensor
        if it["transform"] == "logpos":
            z = torch.log(val.clamp_min(1e-30))
        else:
            z = val
        zs.append(z.reshape(-1))
    return torch.cat(zs).cpu().numpy()


@torch.no_grad()
def set_from_z(gp, items, z_np: np.ndarray):
    """Set params from vector z (inverse transform)."""
    z = torch.as_tensor(z_np, dtype=torch.double, device=next(gp.parameters()).device)
    idx = 0
    for it in items:
        n = int(np.prod(it["shape"]))
        zslice = z[idx : idx + n]
        if it["transform"] == "logpos":
            vs = torch.exp(zslice).view(it["shape"])
        else:
            vs = zslice.view(it["shape"])
        it["setter"](vs)
        idx += n
    assert idx == z.numel()


def build_bounds(
    items,
    ls_bounds,
    os_bounds,
    noise_bounds,
    mean_bounds,
    period_bounds,
):
    """
    Build elementwise bounds aligned with z, in z-space (log for positive).
    Returns (lo, hi) as numpy arrays.
    """
    lo_arr, hi_arr = [], []
    for it in items:
        n = int(np.prod(it["shape"]))
        if it["name"] == "lengthscale":
            l, h = ls_bounds
        elif it["name"] == "outputscale":
            l, h = os_bounds
        elif it["name"] == "noise":
            l, h = noise_bounds
        elif it["name"] == "mean_constant":
            l, h = mean_bounds
        elif it["name"] == "period_length":
            l, h = period_bounds
        else:
            # default to wide real
            l, h = -1e6, 1e6
        if it["transform"] == "logpos":
            l, h = math.log(l), math.log(h)
        lo_arr.extend([l] * n)
        hi_arr.extend([h] * n)
    return np.array(lo_arr, float), np.array(hi_arr, float)


# -------------------- CMA objective and optimizer --------------------


def _objective_factory(gp, items, verbose=False):
    def f(z):
        set_from_z(gp, items, z)
        val = -_trainpath_mll(gp)  # minimize -MLL
        if not np.isfinite(val):
            return 1e100
        if verbose:
            print(f"CMA-ES MLL: {-val:.6f}")
        return float(val)

    return f


def _project_z0_to_bounds(
    z0: np.ndarray, lo: np.ndarray, hi: np.ndarray, verbose: bool
) -> np.ndarray:
    """
    Ensure z0 lies strictly inside [lo, hi] (same decision space as CMA)
    """
    assert z0.shape == lo.shape == hi.shape, "z0/lo/hi shape mismatch"
    eps = 1e-12
    z0_proj = np.minimum(np.maximum(z0, lo + eps), hi - eps)

    if verbose:
        below = np.where(z0 < lo)[0]
        above = np.where(z0 > hi)[0]
        n_viol = below.size + above.size
        if n_viol:
            print(
                f"[CMA init] z0 projected into bounds at {n_viol} coord(s). "
                f"(kept strictly inside to satisfy pycma bounds)"
            )
            show = 8
            for i in list(below[:show]):
                print(f"  idx {i}: {z0[i]:.6g} < lo={lo[i]:.6g} -> {z0_proj[i]:.6g}")
            for i in list(above[:show]):
                print(f"  idx {i}: {z0[i]:.6g} > hi={hi[i]:.6g} -> {z0_proj[i]:.6g}")
            if n_viol > show:
                print(f"  ... and {n_viol - show} more")
    return z0_proj


def lbfgsb_refine_with_logging(
    gp,
    maxfun=2000,
    ftol=1e-3,
    gtol=1e-5,
    timeout_sec=1800.0,
    max_attempts=5,
    verbose=True,
):
    """
    Run fit_gpytorch_mll with an iteration callback that logs the current loss.
    On any exception, restore the pre-fit state_dict so you don't lose a working GP.

    Returns
    -------
    logs : list[dict]
        Each entry has: {'step', 'mll', 'status', 'message'}.
    ok : bool
        True if BoTorch reported SUCCESS/STOPPED, False otherwise.
    """
    logger = logging.getLogger("FewShotCVRPLogger")

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    gp.train()
    gp.likelihood.train()

    # Save pre-fit state so we can roll back on failure
    pre_state = deepcopy(mll.state_dict())
    logs = []

    def _cb(params: dict, result):
        # result.fval is the objective value **being minimized** (-MLL)
        entry = {
            "step": int(getattr(result, "step", -1)),
            "mll": float(-result.fval),
            "status": int(getattr(result, "status", OptimizationStatus.RUNNING)),
            "message": getattr(result, "message", None),
        }
        logs.append(entry)
        if entry["step"] % 5 == 0:
            logger.info(
                f"[LBFGSB {entry['step']:04d}] | log likelihood: {entry['mll']:.6f}"
            )
        if verbose and entry["step"] % 5 == 0:
            print(f"[LBFGSB {entry['step']:04d}] | MLL: {entry['mll']:.6f}")

    try:
        # SciPy L-BFGS-B via BoTorch; pass callback + options & limits
        fit_gpytorch_mll(
            mll,
            optimizer_kwargs={
                "method": "L-BFGS-B",
                "options": {
                    "maxfun": maxfun,  # cap function evaluations
                    "ftol": ftol,
                    "gtol": gtol,
                    # "maxiter": 50,  # optional: also cap iterations
                    # "disp": True,  # optional: let SciPy print too
                },
                "callback": _cb,  # log every iteration
                "timeout_sec": timeout_sec,
            },
            max_attempts=max_attempts,  # BoTorch retry policy
        )
        ok = True
    except Exception as e:
        ok = False
        # Roll back to known-good parameters
        mll.load_state_dict(pre_state)
        logger.warning(f"[LBFGSB] Exception caught, state restored. Reason: {e!r}")
        if verbose:
            print(f"[LBFGSB] Exception caught, state restored. Reason: {e!r}")
    finally:
        gp.eval()
        gp.likelihood.eval()

    return logs, ok


class MLLFitConfig(BaseModel):
    """
    MLL fit kwargs. Accepts NEW/EXTRA keys as well (extra='allow')
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    sigma0: Optional[float] = None
    maxfevals_cma: int = 2000
    maxfevals_grad: int = 2000
    grad_timeout_sec: float = 60.0
    tolx_cma: float = 1e-10
    tolfun_cma: float = 1e-3
    tolfunhist_cma: float = 1e-5
    tolfun_grad: float = 1e-5
    tolgrad_grad: float = 1e-6

    ls_bounds: Tuple[float, float] = (1e-3, 1e4)
    os_bounds: Tuple[float, float] = (1e-4, 1e4)
    noise_bounds: Tuple[float, float] = (1e-6, 1e1)
    mean_bounds: Tuple[float, float] = (-1e3, 1e3)
    period_bounds: Tuple[float, float] = (1e-4, 1e4)

    refine_with_grad: bool = True
    init_lengthscales_with_vanilla_bo: bool = False
    verbose: bool = False

    def as_kwargs(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=False)


def optimize_mll(
    gp: Any,
    cfg: MLLFitConfig,
    seed: Optional[int] = None,
):
    """
    1) Pack all present + trainable hypers into z (log for positives).
    2) Maximize train-path MLL using CMA-ES with bounds.
    3) Optionally refine with fit_gpytorch_mll (L-BFGS-B).
    Returns gp (mutated in-place) and a small result dict.
    """

    logger = logging.getLogger("FewShotCVRPLogger")

    gp.train()
    gp.likelihood.train()

    # Initialize lengthscales (once) before packing
    if cfg.init_lengthscales_with_vanilla_bo:
        _init_lengthscales_vanilla_bo_(gp, seed=seed)

    # discover & pack
    items = discover_mll_params(gp)  # includes only requires_grad params
    if not items:
        return gp, {"msg": "No optimizable hypers discovered (requires_grad=False)."}

    z0 = pack_to_z(gp, items)
    lo, hi = build_bounds(
        items,
        cfg.ls_bounds,
        cfg.os_bounds,
        cfg.noise_bounds,
        cfg.mean_bounds,
        cfg.period_bounds,
    )

    # x0 must lie within bounds in the same space (pycma requirement)
    z0 = _project_z0_to_bounds(z0, lo, hi, verbose=cfg.verbose)

    sigma0 = cfg.sigma0
    if cfg.sigma0 is None:
        sigma0 = _sigma0_from_bounds(lo, hi, rule="quarter")

    f = _objective_factory(gp, items, verbose=cfg.verbose)
    logger.info(f"GP fitting initial log likelihood: {-f(z0):.6f}")

    opts = {
        "bounds": [lo.tolist(), hi.tolist()],  # bounds in z-space
        "seed": seed,
        "maxfevals": cfg.maxfevals_cma,
        "verb_log": 0,
        "verbose": -9,
        "tolx": cfg.tolx_cma,
        "tolfun": cfg.tolfun_cma,
        "tolfunhist": cfg.tolfunhist_cma,
    }
    xbest, fbest, *_ = cma.fmin(
        f, z0, sigma0, eval_initial_x=True, options=opts
    )  # runs CMA-ES
    logger.info(f"GP fitting after CMA-ES log likelihood: {-fbest:.6f}")
    if cfg.verbose:
        print(f"CMA-ES MLL at best: {-fbest:.6f}")

    # set best and (optionally) refine with gradient-based fit
    set_from_z(gp, items, xbest)
    if cfg.refine_with_grad:
        if cfg.verbose:
            print("Running LBFGS-B for local refinement")
        logs, ok = lbfgsb_refine_with_logging(
            gp,
            maxfun=cfg.maxfevals_grad,
            ftol=cfg.tolfun_grad,
            gtol=cfg.tolgrad_grad,
            timeout_sec=cfg.grad_timeout_sec,
            max_attempts=5,
            verbose=cfg.verbose,
        )
        if logs:
            last = logs[-1]
            logger = logging.getLogger("FewShotCVRPLogger")
            logger.log(
                logging.INFO,
                f"GP fitting final log likelihood: {last['mll']:.6f} | status={last['status']}",
            )

    gp.eval()
    gp.likelihood.eval()

    return gp, {
        "items": [it["name"] for it in items],
        # "cma_xbest": xbest,
        "best_trainpath_mll": compute_likelihood(gp),
        "best_cma_mll": -fbest,  # CMA objective is -LL
    }
