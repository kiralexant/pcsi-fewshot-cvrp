"""Parallel-friendly evolutionary algorithms leveraging theta controllers."""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from FewShotCVRP.dataset.theta_control_loader import ThetaControlWrapper
from FewShotCVRP.ea import (
    fitness,
    is_permutation,
    mutate_shift_2opt_fast_fast,
    mutate_shift_swap_fast_fast,
)

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

MutationCallable = Callable[[np.ndarray, np.random.Generator, float], np.ndarray]

_CVRP_REFERENCE = None
_MUTATION_KIND_MAP: Dict[str, MutationCallable] = {
    "swap": mutate_shift_swap_fast_fast,
    "2opt": mutate_shift_2opt_fast_fast,
}


def _force_single_thread_env() -> None:
    for var in _THREAD_ENV_VARS:
        os.environ[var] = "1"
    os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
    os.environ.setdefault("KMP_BLOCKTIME", "0")


def _set_torch_single_thread() -> None:
    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass


def _worker_initializer() -> None:
    # _force_single_thread_env()
    # _set_torch_single_thread()
    pass


def _get_mutation_fn(kind: str) -> MutationCallable:
    try:
        return _MUTATION_KIND_MAP[kind]
    except KeyError as err:
        raise ValueError(
            f"Unsupported mutation kind '{kind}'. Expected one of {sorted(_MUTATION_KIND_MAP)}."
        ) from err


def _split_work(total: int, parts: int) -> Iterable[int]:
    if parts <= 0:
        yield total
        return
    base = total // parts
    remainder = total % parts
    for idx in range(parts):
        chunk = base + (1 if idx < remainder else 0)
        if chunk:
            yield chunk


def _evaluate_chunk(
    seed: int,
    parent: np.ndarray,
    theta: float,
    chunk_size: int,
    mutation_kind: str,
) -> Tuple[np.ndarray, float]:
    # _set_torch_single_thread()
    rng = np.random.default_rng(seed)
    mutation_fn = _get_mutation_fn(mutation_kind)
    cvrp = _CVRP_REFERENCE
    if cvrp is None:
        raise RuntimeError("CVRP instance is not initialised in worker process.")

    best_child = None
    best_value = None

    for _ in range(int(chunk_size)):
        child = mutation_fn(parent, rng, theta)
        child_val = fitness(cvrp, child)
        if (best_value is None) or (child_val < best_value):
            best_child = child
            best_value = float(child_val)

    if best_child is None:
        best_child = parent.copy()
        best_value = float(fitness(cvrp, best_child))

    return best_child, float(best_value)


def one_plus_lambda_ea_parallel_nn_control(
    cvrp,
    theta_controller: ThetaControlWrapper,
    *,
    lambda_: int = 100,
    max_evals: int = 100_000,
    mutation: str = "swap",
    init_perm: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
    patience: Optional[int] = None,
    return_routes: bool = False,
    verbose: bool = False,
    generations_number: Optional[int] = None,
    numproc: int = 1,
) -> Dict[str, Any]:
    """
    (1+λ) EA variant that pulls theta decisions from :class:`ThetaControlWrapper`.

    The controller encapsulates both the bootstrap schedule and the learned
    theta policy. During the warm-up phase the controller yields values from
    the schedule stored in the descriptor; afterwards it feeds the underlying
    neural network with the recorded generation history.

    ``mutation`` selects the mutation operator: ``"swap"`` uses
    :func:`FewShotCVRP.ea.mutate_shift_swap_fast_fast`, ``"2opt"`` uses
    :func:`FewShotCVRP.ea.mutate_shift_2opt_fast_fast`.

    When ``numproc > 1`` the offspring evaluations are split across child
    processes to reduce wall-clock time. Every worker receives a deterministic
    RNG seed derived from ``seed`` ensuring reproducible runs.
    """

    if generations_number is None:
        raise ValueError("Provide generations_number to control the evaluation budget.")
    if lambda_ <= 0:
        raise ValueError("lambda_ must be positive.")

    theta_controller.reset()
    rng = np.random.default_rng(seed)
    n = int(cvrp.n)

    if init_perm is None:
        parent = np.arange(n, dtype=int)
        rng.shuffle(parent)
    else:
        parent = np.array(init_perm, dtype=int)
        if not is_permutation(parent, n):
            raise ValueError("init_perm must be a permutation of 0..n-1")

    best_val = float(fitness(cvrp, parent))
    evals_total = 1
    evals_children = 0
    child_budget = int(min(max_evals, lambda_ * generations_number))

    history = [(evals_total, -1.0, float(best_val))]
    gens_done = 0
    no_improv_gens = 0

    if verbose:
        print(f"[init] fitness={best_val:.6f}")

    mutation_fn = _get_mutation_fn(mutation)
    use_parallel = numproc and numproc > 1

    executor: Optional[ProcessPoolExecutor] = None

    if use_parallel:
        global _CVRP_REFERENCE
        _CVRP_REFERENCE = cvrp
        executor = ProcessPoolExecutor(
            max_workers=numproc,
            mp_context=mp.get_context("fork"),
            initializer=_worker_initializer,
        )

    try:
        for g in range(generations_number):
            if evals_children >= child_budget:
                break
            gens_done += 1

            theta = float(theta_controller.recommend_theta())

            remaining_children = child_budget - evals_children
            if remaining_children <= 0:
                break

            offspring_to_generate = min(lambda_, remaining_children)

            best_child_val = None
            best_child = None

            if use_parallel and executor is not None:
                chunks = list(_split_work(offspring_to_generate, numproc))
                futures = []
                for chunk_size in chunks:
                    seed_child = int(rng.integers(0, 10**9))
                    futures.append(
                        executor.submit(
                            _evaluate_chunk,
                            seed_child,
                            parent.copy(),
                            theta,
                            chunk_size,
                            mutation,
                        )
                    )

                for fut in as_completed(futures):
                    child, child_val = fut.result()
                    if (best_child_val is None) or (child_val < best_child_val):
                        best_child_val = child_val
                        best_child = child

                evals_children += offspring_to_generate
                evals_total += offspring_to_generate

            else:
                for _ in range(offspring_to_generate):
                    child = mutation_fn(parent, rng, theta)
                    child_val = float(fitness(cvrp, child))
                    evals_children += 1
                    evals_total += 1

                    if (best_child_val is None) or (child_val < best_child_val):
                        best_child_val = child_val
                        best_child = child

            if best_child_val is None:
                best_child_val = float(best_val)
                best_child = parent
            else:
                best_child_val = float(best_child_val)

            theta_controller.record_generation(best_val, best_child_val, theta)

            improved = best_child_val < best_val
            if improved:
                parent = best_child
                best_val = best_child_val
                no_improv_gens = 0
            else:
                no_improv_gens += 1

            history.append((evals_total, theta, float(best_val)))

            if verbose:
                tag = "↑" if improved else "·"
                print(
                    f"[gen {gens_done}] theta={theta:.3g} fitness={best_val:.6f} {tag} "
                    f"evals={evals_total}/{1+child_budget}"
                )

            if patience is not None and no_improv_gens >= patience:
                if verbose:
                    print(
                        f"[stop] patience reached: {patience} generations without improvement"
                    )
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if use_parallel:
            globals()["_CVRP_REFERENCE"] = None

    result: Dict[str, Any] = {
        "best_perm": parent,
        "best_fitness": float(best_val),
        "evals": evals_total,
        "gens": gens_done,
        "history": history,
    }

    if return_routes:
        _, routes = cvrp.dynamic_programming_fitness_and_ans(parent)
        result["routes"] = routes

    return result
