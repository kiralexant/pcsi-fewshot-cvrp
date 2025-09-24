"""Utilities for working with per-instance descriptor metadata."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch

import FewShotCVRP.bo.bo_torch as bo_torch
from FewShotCVRP.dataset import parse_and_optimize as pao
import FewShotCVRP.nn.fnn as fnn
from FewShotCVRP.utils.fixed_queue import NumpyFixedQueue


@dataclass(frozen=True)
class ModelDescriptor:
    """Model metadata paired with the CVRP instance it was trained on."""

    instance_name: str
    snapshot_path: Path
    instance_path: Path
    nn_training_method: str


class ThetaControlWrapper:
    """Stateful helper around a trained theta control network."""

    def __init__(
        self,
        theta_function: Callable[[np.ndarray], float],
        window_size: int,
        input_dim: int,
        theta_schedule_window: Iterable[float],
    ) -> None:
        self._theta_function = theta_function
        self._window_size = int(window_size)
        self._input_dim = int(input_dim)
        self._history = NumpyFixedQueue(maxlen=self._input_dim)
        self._theta_schedule_window = [float(v) for v in theta_schedule_window]

        if len(self._theta_schedule_window) != self._window_size:
            raise ValueError(
                "theta_schedule_window length must match window_size to cover initial generations."
            )

        if self._input_dim != 2 * self._window_size:
            raise ValueError(
                "Expected input_dim to be exactly twice the window_size to store "
                "(fitness, theta) pairs."
            )

    @staticmethod
    def compute_fitness_feature(previous_fitness: float, candidate_fitness: float) -> float:
        """Convert two fitness values into a normalized NN input feature."""
        return float(fnn.nn_input_fitness(previous_fitness, candidate_fitness))

    @staticmethod
    def compute_theta_feature(theta: float) -> float:
        """Convert a theta value into a normalized NN input feature."""
        return float(fnn.nn_input_theta(theta))

    def record_generation(
        self,
        prv_fitness: float,
        cur_fitness: float,
        theta: float,
        *,
        fitness_is_normalized: bool = False,
        theta_is_normalized: bool = False,
    ) -> None:
        """
        Record the fitness/parameter state for one EA generation.

        Parameters
        ----------
        prv_fitness:
            Fitness of the parent solution from the previous generation.
        cur_fitness:
            Fitness of the best offspring produced in the current generation.
        theta:
            Either the raw mutation control parameter or its normalized feature
            depending on ``theta_is_normalized``.
        fitness_is_normalized:
            When ``True`` the provided ``cur_fitness`` is assumed to already be
            a normalized feature.
        theta_is_normalized:
            When ``True`` the provided ``theta`` value is assumed to already be
            normalized via :meth:`compute_theta_feature`.
        """

        if fitness_is_normalized:
            fitness_feature = float(cur_fitness)
        else:
            fitness_feature = self.compute_fitness_feature(prv_fitness, cur_fitness)

        theta_value = float(theta)
        if not theta_is_normalized:
            theta_value = self.compute_theta_feature(theta_value)

        self._history.append(fitness_feature)
        self._history.append(theta_value)

    def is_ready(self) -> bool:
        """Return ``True`` once enough history is gathered for predictions."""
        return len(self._history) == self._input_dim

    def theta_schedule_window(self) -> List[float]:
        """Return the bootstrap theta schedule from the descriptor."""
        return list(self._theta_schedule_window)

    def recommend_theta(self) -> float:
        """Return the recommended theta based on the stored history."""
        if not self.is_ready():
            schedule_index = len(self._history) // 2
            if schedule_index < len(self._theta_schedule_window):
                return float(self._theta_schedule_window[schedule_index])
            raise ValueError(
                "Not enough (fitness, theta) pairs collected and schedule exhausted."
            )
        args = self._history.to_numpy(copy=False)
        return float(self._theta_function(args))

    def reset(self) -> None:
        """Clear the stored history."""
        self._history.clear()


class DescriptorRepository:
    """Load CVRP instances and neural controllers from a descriptor."""

    def __init__(self, descriptor_path: Path | str) -> None:
        self._descriptor_path = Path(descriptor_path)
        self._descriptor_dir = self._descriptor_path.parent
        self._descriptor = self._load_descriptor()

        self._nn_cfg = self._descriptor.get("nn", {})
        self._ea_cfg = self._descriptor.get("ea", {})
        self._activation_fn = self._resolve_activation(self._nn_cfg.get("activation", "SiLU"))

        self._dataset_root = Path(pao.__file__).resolve().parent
        self._models = self._load_model_descriptors()
        self._instances = self._load_instances()

    @property
    def models(self) -> List[ModelDescriptor]:
        return list(self._models)

    @property
    def instances(self) -> List[pao.ProblemInstance]:
        return list(self._instances.values())

    @property
    def nn_config(self) -> Dict[str, object]:
        return dict(self._nn_cfg)

    @property
    def ea_config(self) -> Dict[str, object]:
        return dict(self._ea_cfg)

    def get_closest_instance(self, n: int) -> pao.ProblemInstance:
        """Return the problem instance with the closest number of vertices."""
        if not self._instances:
            raise ValueError("No problem instances loaded from descriptor.")
        return min(self._instances.values(), key=lambda inst: abs(inst.n - n))

    def create_theta_controller(self, instance: pao.ProblemInstance) -> ThetaControlWrapper:
        """Build a theta controller wrapper for the given instance."""
        entry = self._find_model_for_instance(instance)
        theta_fun = self._build_theta_function(entry)
        window_size = int(self._ea_cfg.get("ea_window_size", 5))
        input_dim = int(self._nn_cfg.get("in_dim", 2 * window_size))
        schedule = self._ea_cfg.get("theta_schedule_window")
        if schedule is None:
            raise ValueError(
                "Descriptor is missing 'theta_schedule_window' required for bootstrap stage."
            )
        return ThetaControlWrapper(
            theta_fun,
            window_size=window_size,
            input_dim=input_dim,
            theta_schedule_window=schedule,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_descriptor(self) -> Dict[str, object]:
        with self._descriptor_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _resolve_activation(self, activation_name: str) -> Callable[[], torch.nn.Module]:
        try:
            activation_cls = getattr(torch.nn, activation_name)
        except AttributeError as exc:
            raise ValueError(f"Unknown activation '{activation_name}' in descriptor.") from exc

        if not callable(activation_cls):
            raise ValueError(
                f"Activation '{activation_name}' is not callable and cannot be used."
            )

        def factory() -> torch.nn.Module:
            return activation_cls()

        return factory

    def _load_model_descriptors(self) -> List[ModelDescriptor]:
        models_raw = self._descriptor.get("models", [])
        models: List[ModelDescriptor] = []
        for raw in models_raw:
            snapshot_path = self._resolve_path(raw.get("snapshot_path"))
            instance_path = self._resolve_path(
                raw.get("nn_trained_on_instance"),
                fallback_root=self._dataset_root,
            )
            instance_name = instance_path.stem
            models.append(
                ModelDescriptor(
                    instance_name=instance_name,
                    snapshot_path=snapshot_path,
                    instance_path=instance_path,
                    nn_training_method=str(raw.get("nn_training_method", "")),
                )
            )
        return models

    def _resolve_path(self, relative: Optional[str], fallback_root: Optional[Path] = None) -> Path:
        if relative is None:
            raise ValueError("Descriptor entry is missing a required path.")
        primary = (self._descriptor_dir / relative).resolve()
        if primary.exists():
            return primary
        if fallback_root is not None:
            fallback = (fallback_root / relative).resolve()
            if fallback.exists():
                return fallback
        raise FileNotFoundError(
            f"Unable to resolve path '{relative}' relative to '{self._descriptor_dir}'."
        )

    def _load_instances(self) -> Dict[str, pao.ProblemInstance]:
        instances: Dict[str, pao.ProblemInstance] = {}
        for model in self._models:
            if model.instance_name in instances:
                continue
            xml_tree = pao.ET.parse(model.instance_path)
            instance = pao.ProblemInstance(xml_tree)
            instances[model.instance_name] = instance
        return instances

    def _find_model_for_instance(self, instance: pao.ProblemInstance) -> ModelDescriptor:
        entry = next(
            (model for model in self._models if model.instance_name == instance.name),
            None,
        )
        if entry is None:
            raise ValueError(
                f"No model metadata found for instance '{instance.name}'."
            )
        return entry

    def _build_theta_function(self, entry: ModelDescriptor) -> Callable[[np.ndarray], float]:
        snapshot = bo_torch.BayesianOptimizer.load_snapshot(str(entry.snapshot_path))
        hidden_dims_cfg = self._nn_cfg.get("hidden_dims", [])
        hidden_dims: Iterable[int]
        if hidden_dims_cfg is None:
            hidden_dims = []
        else:
            hidden_dims = list(hidden_dims_cfg)

        objective = fnn.NNObjective(
            theta_min=float(self._nn_cfg.get("theta_min", 1.0)),
            theta_max=float(self._nn_cfg.get("theta_max", 100.0)),
            in_dim=int(self._nn_cfg.get("in_dim", 2 * int(self._ea_cfg.get("ea_window_size", 5)))),
            hidden_dims=hidden_dims,
            activation=self._activation_fn,
        )
        flat = np.asarray(snapshot.result["x_rec_mean_in_data"])
        fnn.set_flat_params_(objective.net, flat)
        return objective.build_function()
