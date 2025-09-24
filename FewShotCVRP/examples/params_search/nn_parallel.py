"""
Reimplemented nn.py
-------------------
- Keeps BoTorch (main process) threaded.
- Runs simulations in *child processes* that force single-threaded NumPy/BLAS & PyTorch,
  *without* touching the main process threading.
- Avoids importing NumPy / Torch / project modules at top-level so children can set
  env vars before those imports.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ----------------------------
# Per-process threading control
# ----------------------------
THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
]


def _force_single_thread_env() -> None:
    """Force BLAS/OpenMP single-thread *in this process* before importing numpy/torch."""
    for k in THREAD_ENV_VARS:
        os.environ[k] = "1"
    # Optional Intel MKL knobs to reduce oversubscription noise
    os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
    os.environ.setdefault("KMP_BLOCKTIME", "0")


# ----------------------------
# Child worker
# ----------------------------
@dataclass
class SimulationConstantsLite:
    """Thin transport struct; mirrors simulation.SimulationConstants"""

    cvrp_instance_str: str
    ea_window_size: int
    ea_generations_number: int
    ea_lambda_: int
    ea_mutation_operator_dotted: str  # store dotted path, resolve in child

    in_dim: int
    hidden_dims: Iterable[int]
    theta_min: float
    theta_max: float
    activation_dotted: str  # store dotted path, resolve in child


def _resolve_dotted_attr(path: str):
    import importlib

    module_name, _, attr = path.rpartition(".")
    mod = importlib.import_module(module_name)
    return getattr(mod, attr)


def _simulate_one(
    weights_flat: List[float], cfg: SimulationConstantsLite, seed: int
) -> float:
    """
    Child process entry point. Sets env to single-thread and imports heavy libs afterwards.
    """
    _force_single_thread_env()

    # Heavy imports *after* env is set:
    import numpy as np
    import torch

    # Single-thread torch inside child
    try:
        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass

    # Resolve project modules after env is set
    from pathlib import Path as _Path

    import cvrp_cpp as cc
    import FewShotCVRP.dataset.parse_and_optimize as pao
    import FewShotCVRP.ea as ea
    import FewShotCVRP.examples.params_search.simulation as simulation
    import FewShotCVRP.nn.fnn as fnn

    # Rebuild SimulationConstants-compatible objects
    # Resolve activation / mutation from dotted paths
    activation_cls = _resolve_dotted_attr(cfg.activation_dotted)
    mutation_fun = _resolve_dotted_attr(cfg.ea_mutation_operator_dotted)

    # Build CVRP instance (same as simulation.get_cvrp_instance)
    dataset_dir = _Path(pao.__file__).resolve().parent
    instance = pao.ProblemInstance(pao.ET.parse(dataset_dir / cfg.cvrp_instance_str))
    cvrp = cc.CVRP(
        np.asarray(instance.depot_dist),
        np.asarray(instance.dist),
        np.asarray(instance.node_q),
        np.asarray(instance.capacity),
    )

    # Build objective
    objective = fnn.NNObjective(
        theta_min=float(cfg.theta_min),
        theta_max=float(cfg.theta_max),
        in_dim=int(cfg.in_dim),
        hidden_dims=tuple(int(h) for h in cfg.hidden_dims),
        dtype=torch.float64,  # match BoTorch default
    )

    def param_control_wrapper(args, f):
        assert len(args) == 2 * int(cfg.ea_window_size)
        return f(args)

    # EA simulation (mirrors simulation._worker_fnn)
    simulate = lambda predict_theta: ea.one_plus_lambda_ea_with_theta_control(
        cvrp,
        theta_schedule_window=[100.0, 100.0, 50.0, 50.0, 20.0],
        window=int(cfg.ea_window_size),
        theta_control_fun=lambda args: param_control_wrapper(args, predict_theta),
        seed=int(seed),
        lambda_=int(cfg.ea_lambda_),
        max_evals=int(cfg.ea_generations_number) * int(cfg.ea_lambda_),
        generations_number=int(cfg.ea_generations_number),
        mutation=mutation_fun,
        verbose=False,
    )["best_fitness"]

    # Evaluate NN with given weights
    return float(objective(np.asarray(weights_flat, dtype=np.float64), simulate))


def evaluate_candidates_parallel(
    candidates: Sequence[Sequence[float]],
    sim_cfg: SimulationConstantsLite,
    rng_seed: int,
    max_workers: Optional[int] = None,
) -> List[float]:
    """Evaluate candidate weight vectors in parallel child processes."""

    # _force_multi_thread_env(intra=os.cpu_count() or 8, inter=None)
    # spawn is safer for OpenMP/MKL
    ctx = mp.get_context("spawn")
    results: List[float] = [float("inf")] * len(candidates)
    # independent RNG for child seeds
    import numpy as _np

    local_rng = _np.random.default_rng(int(rng_seed))

    with ProcessPoolExecutor(
        max_workers=max_workers or (os.cpu_count() or 1), mp_context=ctx
    ) as ex:
        fut2idx = {}
        for i, w in enumerate(candidates):
            seed = int(local_rng.integers(0, 10**9))
            fut = ex.submit(_simulate_one, list(map(float, w)), sim_cfg, seed)
            fut2idx[fut] = i
        for fut in as_completed(fut2idx):
            i = fut2idx[fut]
            results[i] = float(fut.result())

    # Re-assert main process threading in case anything toggled it
    # _force_multi_thread_env(intra=os.cpu_count() or 8, inter=None)
    return results


# ----------------------------
# BO wrapper with snapshots
# ----------------------------
def make_BOSaveSnapshots(bo_torch_module, config_loader):
    @dataclass
    class BOSaveSnapshots(bo_torch_module.BayesianOptimizer):
        cur_root_path: str = ""
        bo_start_time: str = ""
        cvrp_instance_str: str = ""
        experiment_config: Dict[str, Any] = None
        logger: Optional[logging.Logger] = None

        def __post_init__(self):
            super().__post_init__()
            self.bo_iteration_number = 0
            config_loader.function_to_dotted_attr(self.experiment_config)

        def save_snapshot(self, save_dir: str, results: Dict = None) -> Path:
            d = super().save_snapshot(save_dir=save_dir, results=results)
            (d / "experiment-config.json").write_text(
                json.dumps(self.experiment_config, indent=2), encoding="utf-8"
            )
            return d

        def step(self) -> Tuple[Any, Any]:
            super().step()
            self.save_snapshot(
                Path(self.cur_root_path)
                / "outputs"
                / f"{self.bo_start_time}"
                / "per-instance-param-control"
                / f"{self.cvrp_instance_str}"
                / f"bo-{self.bo_iteration_number}"
            )
            best_so_far = float(
                self.y_.min().item() if self.minimize else self.y_.max().item()
            )
            self.logger.info(f"Best-so-far: {int(best_so_far)}")
            self.bo_iteration_number += 1

        def save_final_snapshot(self, results):
            self.save_snapshot(
                Path(self.cur_root_path)
                / "outputs"
                / f"{self.bo_start_time}"
                / "per-instance-param-control"
                / f"{self.cvrp_instance_str}"
                / "bo-final",
                results=results,
            )

    return BOSaveSnapshots


# ----------------------------
# Main
# ----------------------------
def main():
    # Lazy imports to avoid importing numpy/torch at child import time
    from pathlib import Path

    import joblib
    import numpy as np

    import FewShotCVRP.bo.bo_torch as bo_torch
    import FewShotCVRP.bo.gp_fitting as gp_fitting

    # Project imports (these may import torch; okay in the main process)
    import FewShotCVRP.examples.params_search.configs_loader as config_loader
    import FewShotCVRP.nn.fnn as fnn
    from FewShotCVRP.utils.logs import configure_logger

    logger = configure_logger("FewShotCVRPLogger")
    logger.log(level=logging.INFO, msg=f"Started module: {__file__}")
    cfg = config_loader.load_experiment_config(
        Path(__file__).with_name("nn-experiment-config.json")
    )
    random_seed = int(cfg["random_seed"])

    # Paths & naming behavior
    cur_root_path_cfg = cfg.get("paths", {}).get("cur_root_path")
    cur_root_path = (
        Path(cur_root_path_cfg)
        if cur_root_path_cfg
        else Path(__file__).resolve().parent.parent
    )
    strip_xml_ext = bool(cfg.get("paths", {}).get("strip_xml_extension", True))

    # Load precomputed DoEs if needed
    precomputed = None
    bo_cfg = cfg["bo"]
    if str(bo_cfg.get("doe_method")) == "precomputed":
        precomputed = joblib.load(
            cur_root_path / "params_search" / "precomputed_DoEs.joblib"
        )

    # Iterate over CVRP instances
    for cvrp_instance_str in cfg["cvrp_instances"]:
        logger.log(
            level=logging.INFO,
            msg=f"Starting to learn policy for instance: {cvrp_instance_str}",
        )
        simulation_rng = np.random.default_rng(random_seed)

        # --- Simulation constants from config ---
        sim = cfg["simulation"]

        # Store dotted paths instead of callables for child import safety
        # (configs_loader already resolves, so convert back to dotted for transport)
        def _to_dotted(x):
            return f"{x.__module__}.{x.__name__}" if callable(x) else str(x)

        sim_cfg_lite = SimulationConstantsLite(
            cvrp_instance_str=cvrp_instance_str,
            ea_window_size=int(sim["ea_window_size"]),
            ea_generations_number=int(sim["ea_generations_number"]),
            ea_lambda_=int(sim["ea_lambda"]),
            ea_mutation_operator_dotted=_to_dotted(sim["ea_mutation_operator"]),
            in_dim=int(sim["in_dim"]),
            hidden_dims=tuple(int(h) for h in sim["hidden_dims"]),
            theta_min=float(sim["theta_min"]),
            theta_max=float(sim["theta_max"]),
            activation_dotted=_to_dotted(sim["activation"]),
        )

        # --- Objective interface and bounds ---
        objective = fnn.NNObjective(
            theta_min=sim_cfg_lite.theta_min,
            theta_max=sim_cfg_lite.theta_max,
            in_dim=sim_cfg_lite.in_dim,
            hidden_dims=sim_cfg_lite.hidden_dims,
        )
        dim = fnn.n_params(objective.net)
        lb, ub = fnn.make_bounds_for_linear_mlp(objective.net)

        # --- Our parallel batch evaluator using child-per-process single-thread env ---
        num_procs = int(cfg.get("parallel", {}).get("num_procs", 1))

        def f_batch(nn_params: np.ndarray) -> np.ndarray:
            return np.asarray(
                evaluate_candidates_parallel(
                    candidates=nn_params,
                    sim_cfg=sim_cfg_lite,
                    rng_seed=simulation_rng.integers(0, 10**9),
                    max_workers=num_procs,
                ),
                dtype=np.float64,
            )

        # --- BO parameters & initial data ---
        bo_cfg = dict(cfg["bo"])  # shallow copy
        X_init, y_init = None, None
        if str(bo_cfg.get("doe_method")) == "precomputed":
            arrays = precomputed[cvrp_instance_str]
            X_init = arrays["X_"]
            y_init = arrays["y_"]

        # Extract and validate MLL config
        mll_fit_cfg = gp_fitting.MLLFitConfig.model_validate(bo_cfg["mll_fit_config"])

        # Purge keys we set explicitly in constructor
        for k in [
            "f_batch",
            "bounds",
            "X_init",
            "y_init",
            "random_state",
            "cur_root_path",
            "bo_start_time",
            "cvrp_instance_str",
            "experiment_config",
            "mll_fit_config",
        ]:
            bo_cfg.pop(k, None)

        # Create BO subclass after importing bo_torch
        BOSaveSnapshots = make_BOSaveSnapshots(bo_torch, config_loader)

        # --- Create and run BO ---
        bo = BOSaveSnapshots(
            None,
            f_batch=f_batch,
            bounds=list(zip(lb, ub)),
            X_init=X_init,
            y_init=y_init,
            random_state=random_seed,
            cur_root_path=str(cur_root_path),
            bo_start_time=datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss"),
            cvrp_instance_str=(
                os.path.splitext(cvrp_instance_str)[0]
                if strip_xml_ext
                else cvrp_instance_str
            ),
            experiment_config=cfg,
            mll_fit_config=mll_fit_cfg,
            **bo_cfg,
        )

        result = bo.run()
        gp = bo.get_gp()
        print("Best x:", result["x_obs_best"], "Best y:", result["y_obs_best"])
        print("\nARD report:")
        bo.report_ard()
        bo.save_final_snapshot(result)


if __name__ == "__main__":
    main()
