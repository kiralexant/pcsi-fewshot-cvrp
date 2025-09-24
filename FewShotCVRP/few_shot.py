"""Few-shot orchestration utilities for CVRP optimisation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import cvrp_cpp as cc
import FewShotCVRP.ea as ea
import FewShotCVRP.ea_parallel as ea_parallel
from FewShotCVRP.dataset import parse_and_optimize as pao
from FewShotCVRP.dataset.theta_control_loader import (
    DescriptorRepository,
    ThetaControlWrapper,
)


DESCRIPTOR_PATH = (
    Path(__file__).resolve().parent
    / "dataset"
    / "trained_models"
    / "per_instance"
    / "descriptor.json"
)


class ScaledThetaControlWrapper(ThetaControlWrapper):
    """Theta controller that rescales NN recommendations for new instance sizes."""

    def __init__(self, base: ThetaControlWrapper, scale: float) -> None:
        super().__init__(
            theta_function=getattr(base, "_theta_function"),
            window_size=int(getattr(base, "_window_size")),
            input_dim=int(getattr(base, "_input_dim")),
            theta_schedule_window=base.theta_schedule_window(),
        )
        self._scale = float(scale)

    def recommend_theta(self) -> float:
        return float(super().recommend_theta() * self._scale)


@dataclass
class FewShotResult:
    ea_result: Dict[str, object]
    descent_result: Optional[Dict[str, object]]

    @property
    def best(self) -> Dict[str, object]:
        if self.descent_result and (
            self.descent_result.get("best_fitness", float("inf"))
            < self.ea_result.get("best_fitness", float("inf"))
        ):
            return self.descent_result
        return self.ea_result


def _load_problem_instance(instance_name: str) -> pao.ProblemInstance:
    dataset_dir = Path(pao.__file__).resolve().parent
    tree = pao.ET.parse(dataset_dir / instance_name)
    return pao.ProblemInstance(tree)


def _make_cvrp(instance: pao.ProblemInstance) -> cc.CVRP:
    return cc.CVRP(
        np.asarray(instance.depot_dist),
        np.asarray(instance.dist),
        np.asarray(instance.node_q),
        np.asarray(instance.capacity),
    )


def few_shot_optimization(
    instance_name: str,
    random_seed: int | None = None,
    numproc: int = 1,
    *,
    subsequent_local_opt: bool = False,
    local_opt_budget_fraction: float = 0.05,
) -> FewShotResult:
    repo = DescriptorRepository(DESCRIPTOR_PATH)

    target_instance = _load_problem_instance(instance_name)
    target_cvrp = _make_cvrp(target_instance)

    source_instance = repo.get_closest_instance(target_instance.n)
    base_controller = repo.create_theta_controller(source_instance)

    scale = float(target_instance.n) / float(source_instance.n)
    theta_controller = ScaledThetaControlWrapper(base_controller, scale=scale)

    ea_cfg = repo.ea_config
    generations = int(ea_cfg["ea_generations_number"])
    lambda_ = int(ea_cfg["ea_lambda_"])
    max_evals = generations * lambda_

    ea_result = ea_parallel.one_plus_lambda_ea_parallel_nn_control(
        target_cvrp,
        theta_controller,
        lambda_=lambda_,
        max_evals=max_evals,
        generations_number=generations,
        seed=random_seed,
        mutation="2opt",
        numproc=numproc,
        return_routes=True,
        verbose=False,
    )

    descent_result: Optional[Dict[str, float]] = None
    if subsequent_local_opt:
        budget = max(1, int(local_opt_budget_fraction * max_evals))
        descent_result = ea.descent(
            target_cvrp,
            max_evals=budget,
            mutation=ea.mutate_shift_2opt_fast_fast,
            init_perm=ea_result["best_perm"],
            seed=random_seed,
            theta=1.0,
            return_routes=True,
            verbose=False,
        )

    return FewShotResult(ea_result=ea_result, descent_result=descent_result)
