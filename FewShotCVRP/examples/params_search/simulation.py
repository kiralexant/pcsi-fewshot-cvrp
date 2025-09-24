import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch.nn as nn

import cvrp_cpp as cc
import FewShotCVRP.dataset.parse_and_optimize as pao
import FewShotCVRP.ea as ea
import FewShotCVRP.nn.fnn as fnn


def get_cvrp_instance(instance_name_str):
    dataset_dir = Path(pao.__file__).resolve().parent
    instance = pao.ProblemInstance(pao.ET.parse(dataset_dir / instance_name_str))
    return cc.CVRP(
        np.asarray(instance.depot_dist),
        np.asarray(instance.dist),
        np.asarray(instance.node_q),
        np.asarray(instance.capacity),
    )


@dataclass
class SimulationConstants:
    cvrp_instance_str: str

    # ---------- ea parameters ----------
    ea_window_size: int
    ea_generations_number: int
    ea_lambda_: int
    ea_mutation_operator: Callable[[np.ndarray], np.ndarray]

    # ---------- nn parameters ----------
    in_dim: int
    hidden_dims: Iterable[int]
    theta_min: float
    theta_max: float
    activation: Callable[[], nn.Module]


def _worker_fnn(
    nn_weights: np.ndarray, constants: SimulationConstants, simulation_seed: int
):
    cvrp = get_cvrp_instance(constants.cvrp_instance_str)
    objective = fnn.NNObjective(
        theta_min=constants.theta_min,
        theta_max=constants.theta_max,
        in_dim=constants.in_dim,
        hidden_dims=constants.hidden_dims,
    )

    def param_control_wrapper(args, f):
        assert len(args) == 2 * constants.ea_window_size
        return f(args)

    simulation = lambda predict_theta: ea.one_plus_lambda_ea_with_theta_control(
        cvrp,
        theta_schedule_window=[100.0, 100.0, 50.0, 50.0, 20.0],
        window=constants.ea_window_size,
        theta_control_fun=lambda args: param_control_wrapper(args, predict_theta),
        seed=simulation_seed,
        lambda_=constants.ea_lambda_,
        max_evals=constants.ea_generations_number * constants.ea_lambda_,
        generations_number=constants.ea_generations_number,
        mutation=ea.mutate_shift_2opt_fast_fast,
        verbose=False,
    )["best_fitness"]
    return objective(nn_weights, simulation)


def run_parallel_fnn(
    nn_weights_array: np.ndarray,
    constants: SimulationConstants,
    simulation_rng: np.random.Generator,
    num_procs: int | None = None,
):
    results = np.ones(len(nn_weights_array)) * np.inf

    # На Linux лучше 'spawn', на Windows он и так по умолчанию
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass  # уже установлен — ок

    n_workers = num_procs or (os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        fut2idx = {
            ex.submit(
                _worker_fnn, nn_weights, constants, simulation_rng.integers(0, 10**9)
            ): i
            for i, nn_weights in enumerate(nn_weights_array)
        }
        for fut in as_completed(fut2idx):
            i = fut2idx[fut]
            results[i] = fut.result()  # пробросит исключение, если было

    return results
