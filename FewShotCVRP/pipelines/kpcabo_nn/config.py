from __future__ import annotations

import json
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

CallableLike = Union[str, Callable[..., Any]]


def _resolve_dotted_attr(path: str) -> Any:
    if not isinstance(path, str) or "." not in path:
        raise ValueError(f"Expected dotted path string, got: {path!r}")
    module_name, _, attr = path.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def ensure_callable(value: CallableLike) -> Callable[..., Any]:
    if isinstance(value, str):
        return _resolve_dotted_attr(value)
    if callable(value):
        return value
    raise TypeError(f"Expected callable or dotted path, got: {value!r}")


def to_dotted_path(value: CallableLike) -> str:
    if isinstance(value, str):
        return value
    if callable(value):
        module = getattr(value, "__module__", None)
        name = getattr(value, "__name__", None)
        if not module or not name:
            raise ValueError(
                "Cannot infer dotted path for callable without __module__ / __name__"
            )
        return f"{module}.{name}"
    raise TypeError(f"Expected callable or dotted path, got: {value!r}")


@dataclass(frozen=True)
class SimulationConfig:
    window_size: int
    generations_number: int
    lambda_: int
    mutation_operator: CallableLike
    in_dim: int
    hidden_dims: Sequence[int]
    theta_min: float
    theta_max: float
    activation: CallableLike

    @property
    def activation_factory(self) -> Callable[..., Any]:
        return ensure_callable(self.activation)

    @property
    def mutation_dotted(self) -> str:
        return to_dotted_path(self.mutation_operator)

    @property
    def activation_dotted(self) -> str:
        return to_dotted_path(self.activation)

    def as_mapping(self) -> Dict[str, Any]:
        return {
            "ea_window_size": int(self.window_size),
            "ea_generations_number": int(self.generations_number),
            "ea_lambda": int(self.lambda_),
            "ea_mutation_operator": self.mutation_operator,
            "in_dim": int(self.in_dim),
            "hidden_dims": [int(h) for h in self.hidden_dims],
            "theta_min": float(self.theta_min),
            "theta_max": float(self.theta_max),
            "activation": self.activation,
        }


@dataclass(frozen=True)
class ParallelConfig:
    num_procs: Optional[int] = None


@dataclass(frozen=True)
class PathsConfig:
    cur_root_path: Optional[Path] = None
    strip_instance_extension: bool = True
    results_subdir: str = "per-instance-param-control"
    precomputed_doe_filename: str = "precomputed_DoEs.joblib"


@dataclass(frozen=True)
class TrainingConfig:
    instance_names: Sequence[str]
    random_seed: Optional[int]
    simulation: SimulationConfig
    parallel: ParallelConfig
    kpcabo: Mapping[str, Any]
    bo_embedding: Mapping[str, Any]
    paths: PathsConfig = field(default_factory=PathsConfig)

    def copy_with(self, **updates: Any) -> "TrainingConfig":
        data = {
            "instance_names": self.instance_names,
            "random_seed": self.random_seed,
            "simulation": self.simulation,
            "parallel": self.parallel,
            "kpcabo": dict(self.kpcabo),
            "bo_embedding": dict(self.bo_embedding),
            "paths": self.paths,
        }
        data.update(updates)
        return TrainingConfig(**data)


def _parse_simulation_config(data: Mapping[str, Any]) -> SimulationConfig:
    return SimulationConfig(
        window_size=int(data.get("ea_window_size")),
        generations_number=int(data.get("ea_generations_number")),
        lambda_=int(data.get("ea_lambda")),
        mutation_operator=data.get("ea_mutation_operator"),
        in_dim=int(data.get("in_dim")),
        hidden_dims=tuple(int(h) for h in data.get("hidden_dims", [])),
        theta_min=float(data.get("theta_min")),
        theta_max=float(data.get("theta_max")),
        activation=data.get("activation"),
    )


def _parse_paths_config(data: Mapping[str, Any]) -> PathsConfig:
    root = data.get("cur_root_path")
    return PathsConfig(
        cur_root_path=Path(root) if root else None,
        strip_instance_extension=bool(data.get("strip_xml_extension", True)),
        results_subdir=str(data.get("results_subdir", "per-instance-param-control")),
        precomputed_doe_filename=str(
            data.get("precomputed_doe_filename", "precomputed_DoEs.joblib")
        ),
    )


def load_training_config(config_path: Union[str, Path]) -> TrainingConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw_cfg: Dict[str, Any] = json.load(f)

    instance_names = list(
        raw_cfg.get("cvrp_instances", raw_cfg.get("instances", []))
    )
    if not instance_names:
        raise ValueError(
            "Config must provide 'cvrp_instances' or 'instances' with at least one entry."
        )

    simulation_cfg = _parse_simulation_config(raw_cfg.get("simulation", {}))
    parallel_cfg = ParallelConfig(
        num_procs=(raw_cfg.get("parallel", {}) or {}).get("num_procs")
    )
    paths_cfg = _parse_paths_config(raw_cfg.get("paths", {}))

    kpcabo_cfg = dict(raw_cfg.get("kpcabo", {}))
    bo_embedding_cfg = dict(raw_cfg.get("bo_embedding", {}))

    random_seed = raw_cfg.get("random_seed")
    if random_seed is not None:
        random_seed = int(random_seed)

    return TrainingConfig(
        instance_names=tuple(instance_names),
        random_seed=random_seed,
        simulation=simulation_cfg,
        parallel=parallel_cfg,
        kpcabo=kpcabo_cfg,
        bo_embedding=bo_embedding_cfg,
        paths=paths_cfg,
    )


__all__ = [
    "CallableLike",
    "SimulationConfig",
    "ParallelConfig",
    "PathsConfig",
    "TrainingConfig",
    "load_training_config",
    "ensure_callable",
    "to_dotted_path",
]
