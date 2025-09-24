from __future__ import annotations

# ---------- CPU threading hygiene ----------
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch
from rich.console import Console
from rich.logging import RichHandler

import FewShotCVRP.bo.bo_torch as bo_torch
import FewShotCVRP.ea as ea
import FewShotCVRP.examples.params_search.simulation as simulation
import FewShotCVRP.nn.fnn as fnn
from FewShotCVRP.examples.analysis.ea_stats_store import EAStatsStore
import FewShotCVRP.ea_parallel as ea_parallel
import FewShotCVRP.few_shot as few_shot


# ================================================================
# Config & dataclasses
# ================================================================
@dataclass
class RunConfiguration:
    cvrp_instance_name: str
    algorithm_name: str

    # ---------- EA parameters ----------
    ea_window_size: int
    ea_generations_number: int
    ea_lambda_: int

    # ---------- NN parameters ----------
    in_dim: int
    hidden_dims: Iterable[int]
    theta_min: float
    theta_max: float
    activation: Callable[[], torch.nn.Module] = torch.nn.SiLU

    # ---------- Mutation operator ----------
    ea_mutation_operator: Callable[[np.ndarray], np.ndarray] = (
        ea.mutate_shift_2opt_fast_fast
    )


# ================================================================
# Helpers
# ================================================================
def build_theta_control_fun(
    snapshot_dir: str | Path,
    in_dim: int,
    hidden_dims: Iterable[int],
    theta_min: float,
    theta_max: float,
    activation: Callable[[], torch.nn.Module],
) -> Callable[[np.ndarray], float]:
    snap = bo_torch.BayesianOptimizer.load_snapshot(str(snapshot_dir))
    objective = fnn.NNObjective(
        theta_min=theta_min,
        theta_max=theta_max,
        in_dim=in_dim,
        hidden_dims=list(hidden_dims),
        activation=activation,
    )
    flat = np.asarray(snap.result["x_rec_mean_in_data"])
    fnn.set_flat_params_(objective.net, flat)
    theta_control_fun = objective.build_function()
    return theta_control_fun


def _extract_iteration_log(res: Dict[str, Any]) -> List[Tuple[int, float, float]]:
    """
    Extract a per-iteration log of (evals, theta, best_so_far).
    """
    return res["history"]


def _downsample_rows_for_descent(
    rows: list[tuple[int, float, float]], lambda_: int
) -> list[tuple[int, float, float]]:
    """
    Keep only rows where (evals - 1) % lambda_ == 0.
    Also ensure we keep the final row (best-so-far at the end) if it's not a multiple of lambda_.
    """
    if not rows:
        return rows

    out = []
    seen = set()
    for e, t, b in rows:
        if (e - 1) % lambda_ == 0:
            if e not in seen:
                out.append((int(e), float(t), float(b)))
                seen.add(e)

    # always keep the final row even if not divisible by lambda_
    e_last, t_last, b_last = rows[-1]
    if e_last not in seen:
        out.append((int(e_last), float(t_last), float(b_last)))

    out.sort(key=lambda x: x[0])
    return out


def _run_algorithm_once(
    cfg: RunConfiguration,
    seed: int,
    theta_control_fun: Callable[[np.ndarray], float] | None,
):
    max_evals = cfg.ea_generations_number * cfg.ea_lambda_
    if cfg.algorithm_name == "theta_control":
        res = ea.one_plus_lambda_ea_with_theta_control(
            simulation.get_cvrp_instance(cfg.cvrp_instance_name),
            theta_schedule_window=[100.0, 100.0, 50.0, 50.0, 20.0],
            window=cfg.ea_window_size,
            theta_control_fun=theta_control_fun,
            seed=seed,
            lambda_=cfg.ea_lambda_,
            max_evals=max_evals,
            generations_number=cfg.ea_generations_number,
            mutation=cfg.ea_mutation_operator,
            verbose=False,
        )
    elif cfg.algorithm_name == "few_shot":
        repo = few_shot.DescriptorRepository(few_shot.DESCRIPTOR_PATH)

        target_instance = few_shot._load_problem_instance(cfg.cvrp_instance_name)
        target_cvrp = few_shot._make_cvrp(target_instance)

        source_instance = repo.get_closest_instance(target_instance.n)
        base_controller = repo.create_theta_controller(source_instance)

        scale = float(target_instance.n) / float(source_instance.n)
        theta_controller = few_shot.ScaledThetaControlWrapper(
            base_controller, scale=scale
        )

        res = ea_parallel.one_plus_lambda_ea_parallel_nn_control(
            target_cvrp,
            theta_controller,
            lambda_=cfg.ea_lambda_,
            max_evals=max_evals,
            generations_number=cfg.ea_generations_number,
            seed=seed,
            mutation="2opt",
            numproc=1,
            return_routes=False,
            verbose=False,
        )

    elif cfg.algorithm_name == "descent":
        res = ea.descent(
            cvrp=simulation.get_cvrp_instance(cfg.cvrp_instance_name),
            max_evals=max_evals,
            mutation=cfg.ea_mutation_operator,
            seed=seed,
            verbose=False,
        )
    elif cfg.algorithm_name == "theta_schedule_linear":
        res = ea.one_plus_lambda_ea_with_theta_schedule(
            simulation.get_cvrp_instance(cfg.cvrp_instance_name),
            theta_schedule=np.linspace(100.0, 1.0, cfg.ea_generations_number),
            seed=seed,
            lambda_=cfg.ea_lambda_,
            max_evals=max_evals,
            generations_number=cfg.ea_generations_number,
            mutation=cfg.ea_mutation_operator,
            verbose=False,
        )
    elif cfg.algorithm_name == "theta_fixed_1":
        res = ea.one_plus_lambda_ea_with_theta_schedule(
            simulation.get_cvrp_instance(cfg.cvrp_instance_name),
            theta_schedule=[1.0] * cfg.ea_generations_number,
            seed=seed,
            lambda_=cfg.ea_lambda_,
            max_evals=max_evals,
            generations_number=cfg.ea_generations_number,
            mutation=cfg.ea_mutation_operator,
            verbose=False,
        )
    elif cfg.algorithm_name == "theta_fixed_10":
        res = ea.one_plus_lambda_ea_with_theta_schedule(
            simulation.get_cvrp_instance(cfg.cvrp_instance_name),
            theta_schedule=[10.0] * cfg.ea_generations_number,
            seed=seed,
            lambda_=cfg.ea_lambda_,
            max_evals=max_evals,
            generations_number=cfg.ea_generations_number,
            mutation=cfg.ea_mutation_operator,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown algorithm_name: {cfg.algorithm_name}")

    rows = _extract_iteration_log(res)

    # If descent → keep only every lambda-th eval + final row
    if cfg.algorithm_name == "descent":
        rows = _downsample_rows_for_descent(rows, cfg.ea_lambda_)

    return res, rows


# ================================================================
# Single-process worker
# ================================================================
def _single_process_simulation(
    simulation_seed: int,
    configuration: RunConfiguration,
    snapshot_dir: str | Path | None,
) -> List[Tuple[int, float, float]]:
    try:
        theta_fun = None
        if configuration.algorithm_name == "theta_control":
            if snapshot_dir is None:
                raise ValueError(
                    "theta_control requires 'theta_snapshot_dir' in config."
                )
            theta_fun = build_theta_control_fun(
                snapshot_dir=snapshot_dir,
                in_dim=configuration.in_dim,
                hidden_dims=configuration.hidden_dims,
                theta_min=configuration.theta_min,
                theta_max=configuration.theta_max,
                activation=configuration.activation,
            )
        _, rows = _run_algorithm_once(configuration, simulation_seed, theta_fun)
        return rows
    except Exception as e:
        # Return empty so the main process can continue; log happens in parent.
        return []


# ================================================================
# Parallel launcher
# ================================================================
def run_parallel_experiments(
    num_runs: int,
    configuration: RunConfiguration,
    simulation_rng: np.random.Generator,
    snapshot_dir: str | Path | None,
    num_procs: int | None = None,
) -> List[List[Tuple[int, float, float]]]:
    results: List[List[Tuple[int, float, float]]] = [None] * num_runs  # type: ignore

    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    n_workers = int(num_procs or (os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        fut2idx = {
            ex.submit(
                _single_process_simulation,
                int(simulation_rng.integers(0, 10**9)),
                configuration,
                snapshot_dir,
            ): i
            for i in range(num_runs)
        }
        for fut in as_completed(fut2idx):
            i = fut2idx[fut]
            try:
                results[i] = fut.result()
            except Exception:
                results[i] = []
    return results


# ================================================================
# Main
# ================================================================
def _setup_file_logging() -> logging.Logger:
    """
    Stdout-only Rich logger.
    Note: `log_path` is ignored by design (kept for API compatibility).
    """
    logger = logging.getLogger("ea_runner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers on re-run
    logger.propagate = False

    # Console → stdout (stderr=False ensures stdout)
    console = Console()

    handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,  # cleaner logs
        markup=True,  # enable [bold], [cyan], etc.
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        log_time_format="%Y-%m-%d %H:%M:%S",
    )
    # With RichHandler the handler renders time/level; keep the message clean
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(handler)
    return logger


def _load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _activation_from_name(name: str) -> Callable[[], torch.nn.Module]:
    name = name.lower()
    if name == "silu":
        return torch.nn.SiLU
    if name == "relu":
        return torch.nn.ReLU
    if name == "tanh":
        return torch.nn.Tanh
    raise ValueError(f"Unsupported activation: {name}")


def main():
    # --------- read config ---------
    cur_dir = Path(__file__).resolve().parent
    cfg_path = cur_dir / "ea-runs-config.json"
    cfg = _load_json(cfg_path)

    output_dir = cur_dir / cfg.get("output_dir", "outputs/ea_runs")
    raw_path = output_dir / cfg.get("raw_file", "raw_runs.parquet")
    stats_path = output_dir / cfg.get("stats_file", "aggregated_stats.parquet")

    logger = _setup_file_logging()
    logger.info("Starting EA experiment batch")

    theta_snapshot_dir = cfg.get("theta_snapshot_dir", None)
    if theta_snapshot_dir is not None:
        theta_snapshot_dir = str(cur_dir / theta_snapshot_dir)

    instances: List[str] = cfg["instances"]
    algorithms: List[str] = cfg["algorithms"]
    num_runs_per_combo: int = int(cfg.get("num_runs_per_combo", 1))
    sim_seed: int = int(cfg.get("simulation_rng_seed", 0))
    num_procs: int | None = cfg.get("num_procs", None)

    ea_cfg = cfg["ea"]
    nn_cfg = cfg["nn"]
    activation = _activation_from_name(nn_cfg.get("activation", "SiLU"))

    sim_rng = np.random.default_rng(sim_seed)

    # --------- initialize log store ---------
    store = EAStatsStore()
    if raw_path.exists():
        logger.info(f"Found existing raw file; loading: {raw_path}")
        store.load_raw(raw_path)

    total_combos = len(instances) * len(algorithms)
    logger.info(
        f"Total combos: {total_combos}; runs per combo: {num_runs_per_combo}; procs: {num_procs or os.cpu_count()}"
    )

    for inst in instances:
        for algo in algorithms:
            cfg_run = RunConfiguration(
                cvrp_instance_name=inst,
                algorithm_name=algo,
                ea_window_size=int(ea_cfg["ea_window_size"]),
                ea_generations_number=int(ea_cfg["ea_generations_number"]),
                ea_lambda_=int(ea_cfg["ea_lambda_"]),
                in_dim=int(nn_cfg["in_dim"]),
                hidden_dims=nn_cfg["hidden_dims"],
                theta_min=float(nn_cfg["theta_min"]),
                theta_max=float(nn_cfg["theta_max"]),
                activation=activation,
            )

            logger.info(f"Running combo: algo={algo}, instance={inst}")
            per_run_logs = run_parallel_experiments(
                num_runs=num_runs_per_combo,
                configuration=cfg_run,
                simulation_rng=sim_rng,
                snapshot_dir=theta_snapshot_dir,
                num_procs=num_procs,
            )
            if algo == "theta_control":
                nn_trained_on_instance = cfg["nn_trained_on_instance"]
                nn_training_method = cfg["nn_training_method"]
            else:
                nn_trained_on_instance = None
                nn_training_method = None

            success = 0
            for r_idx, run_rows in enumerate(per_run_logs, start=1):
                if not run_rows:
                    logger.warning(
                        f"  run {r_idx}: no per-iteration data (algo={algo}, inst={inst})"
                    )
                    continue
                store.add_run(
                    data=run_rows,
                    cvrp_instance_name=inst,
                    algorithm_name=algo,
                    meta={"seed": int(sim_rng.integers(0, 10**9))},
                    nn_trained_on_instance=nn_trained_on_instance,
                    nn_training_method=nn_training_method,
                )
                success += 1

            logger.info(
                f"  combo finished: {success}/{num_runs_per_combo} runs recorded"
            )
            # save incrementally (safer for background runs)
            store.save_raw(raw_path)

    store.save_raw(raw_path)
    store.save_stats(stats_path)
    logger.info(f"Done. Raw: {raw_path}")
    logger.info(f"Done. Stats: {stats_path}")


# ----------------------------------------------------------------
if __name__ == "__main__":
    main()
