from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


def build_snapshotting_optimizer(
    bo_module,
    config_loader_module,
    *,
    experiment_subdir: str,
    logger: logging.Logger | None = None,
):
    """Factory that decorates a Bayesian optimizer with snapshotting support."""

    @dataclass
    class SnapshottingOptimizer(bo_module.BayesianOptimizer):
        cur_root_path: str = ""
        run_stamp: str = ""
        instance_name: str = ""
        experiment_config: Dict[str, Any] | None = None
        logger: logging.Logger | None = logger
        experiment_subdir: str = experiment_subdir

        def __post_init__(self) -> None:
            super().__post_init__()
            self._iteration = 0
            if self.experiment_config is None:
                self.experiment_config = {}
            config_loader_module.function_to_dotted_attr(self.experiment_config)

        def _snapshot_dir(self, suffix: str) -> Path:
            return (
                Path(self.cur_root_path)
                / "outputs"
                / self.run_stamp
                / self.experiment_subdir
                / self.instance_name
                / suffix
            )

        def save_snapshot(self, save_dir: str | Path, results: Dict[str, Any] | None = None) -> Path:
            directory = super().save_snapshot(save_dir=save_dir, results=results)
            (directory / "experiment-config.json").write_text(
                json.dumps(self.experiment_config, indent=2), encoding="utf-8"
            )
            return directory

        def step(self):
            res = super().step()
            snapshot_dir = self._snapshot_dir(f"bo-{self._iteration}")
            self.save_snapshot(snapshot_dir)
            if self.logger is not None:
                best = float(self.y_.min().item()) if self.minimize else float(self.y_.max().item())
                self.logger.info("Best-so-far: %d", int(best))
            self._iteration += 1
            return res

        def save_final_snapshot(self, results: Dict[str, Any]) -> None:
            self.save_snapshot(self._snapshot_dir("bo-final"), results=results)

    return SnapshottingOptimizer


__all__ = ["build_snapshotting_optimizer"]
