from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

import FewShotCVRP.dataset.parse_and_optimize as pao
import FewShotCVRP.ea as ea
import FewShotCVRP.nn.fnn as fnn
from .config import SimulationConfig
from .interfaces import BatchEvaluator

THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
]


def _force_single_thread_env() -> None:
    for key in THREAD_ENV_VARS:
        os.environ[key] = "1"
    os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
    os.environ.setdefault("KMP_BLOCKTIME", "0")


@dataclass
class SimulationWorkerConfig:
    instance_name: str
    window_size: int
    generations_number: int
    lambda_: int
    mutation_dotted: str
    in_dim: int
    hidden_dims: Iterable[int]
    theta_min: float
    theta_max: float
    activation_dotted: str


def _resolve_dotted_attr(path: str):
    import importlib

    module_name, _, attr = path.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _simulate_one(weights_flat, cfg: SimulationWorkerConfig, seed: int) -> float:
    _force_single_thread_env()

    import numpy as _np
    import torch as _torch

    try:
        _torch.set_num_threads(1)
        if hasattr(_torch, "set_num_interop_threads"):
            _torch.set_num_interop_threads(1)
    except Exception:
        pass

    from pathlib import Path as _Path

    import cvrp_cpp as cc

    activation_ctor = _resolve_dotted_attr(cfg.activation_dotted)
    mutation_fun = _resolve_dotted_attr(cfg.mutation_dotted)

    dataset_dir = _Path(pao.__file__).resolve().parent
    instance = pao.ProblemInstance(pao.ET.parse(dataset_dir / cfg.instance_name))
    cvrp = cc.CVRP(
        _np.asarray(instance.depot_dist),
        _np.asarray(instance.dist),
        _np.asarray(instance.node_q),
        _np.asarray(instance.capacity),
    )

    objective = fnn.NNObjective(
        theta_min=float(cfg.theta_min),
        theta_max=float(cfg.theta_max),
        in_dim=int(cfg.in_dim),
        hidden_dims=tuple(int(h) for h in cfg.hidden_dims),
        dtype=_torch.float64,
        activation=activation_ctor,
    )

    def param_control_wrapper(args, func):
        assert len(args) == 2 * int(cfg.window_size)
        return func(args)

    simulate = lambda predict_theta: ea.one_plus_lambda_ea_with_theta_control(
        cvrp,
        theta_schedule_window=[100.0, 100.0, 50.0, 50.0, 20.0],
        window=int(cfg.window_size),
        theta_control_fun=lambda args: param_control_wrapper(args, predict_theta),
        seed=int(seed),
        lambda_=int(cfg.lambda_),
        max_evals=int(cfg.generations_number) * int(cfg.lambda_),
        generations_number=int(cfg.generations_number),
        mutation=mutation_fun,
        verbose=False,
    )["best_fitness"]

    return float(objective(_np.asarray(weights_flat, dtype=_np.float64), simulate))


def _evaluate_candidates_parallel(
    candidates: Sequence[Sequence[float]] | np.ndarray,
    sim_cfg: SimulationWorkerConfig,
    rng_seed: int,
    max_workers: int | None,
) -> np.ndarray:
    ctx = mp.get_context("spawn")
    candidates = np.asarray(candidates, dtype=float)
    results = np.full(len(candidates), np.inf, dtype=float)

    import numpy as _np

    local_rng = _np.random.default_rng(int(rng_seed))

    with ProcessPoolExecutor(max_workers=max_workers or (os.cpu_count() or 1), mp_context=ctx) as ex:
        fut2idx = {}
        for idx, weights in enumerate(candidates):
            seed = int(local_rng.integers(0, 10**9))
            fut = ex.submit(_simulate_one, weights.tolist(), sim_cfg, seed)
            fut2idx[fut] = idx
        for fut in as_completed(fut2idx):
            i = fut2idx[fut]
            results[i] = float(fut.result())

    return results


class CVRPFNNBatchEvaluator(BatchEvaluator):
    """Parallel evaluator for CVRP parameter-control neural networks."""

    def __init__(self, instance_name: str, simulation_cfg: SimulationConfig):
        self.instance_name = instance_name
        self.simulation_cfg = simulation_cfg
        activation_ctor = simulation_cfg.activation_factory
        self._objective = fnn.NNObjective(
            theta_min=simulation_cfg.theta_min,
            theta_max=simulation_cfg.theta_max,
            in_dim=simulation_cfg.in_dim,
            hidden_dims=simulation_cfg.hidden_dims,
            dtype=torch.float64,
            activation=activation_ctor,
        )
        self._dimension = fnn.n_params(self._objective.net)
        self._lb, self._ub = fnn.make_bounds_for_linear_mlp(self._objective.net)
        self._worker_cfg = SimulationWorkerConfig(
            instance_name=instance_name,
            window_size=simulation_cfg.window_size,
            generations_number=simulation_cfg.generations_number,
            lambda_=simulation_cfg.lambda_,
            mutation_dotted=simulation_cfg.mutation_dotted,
            in_dim=simulation_cfg.in_dim,
            hidden_dims=simulation_cfg.hidden_dims,
            theta_min=simulation_cfg.theta_min,
            theta_max=simulation_cfg.theta_max,
            activation_dotted=simulation_cfg.activation_dotted,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self._lb, self._ub

    def evaluate_batch(
        self,
        candidates: Sequence[Sequence[float]] | np.ndarray,
        rng_seed: int,
        max_workers: int | None = None,
    ) -> np.ndarray:
        return _evaluate_candidates_parallel(candidates, self._worker_cfg, rng_seed, max_workers)


def make_cvrp_evaluator_factory(simulation_cfg: SimulationConfig) -> Callable[[str], CVRPFNNBatchEvaluator]:
    def factory(instance_name: str) -> CVRPFNNBatchEvaluator:
        return CVRPFNNBatchEvaluator(instance_name, simulation_cfg)

    return factory


__all__ = [
    "CVRPFNNBatchEvaluator",
    "make_cvrp_evaluator_factory",
]
