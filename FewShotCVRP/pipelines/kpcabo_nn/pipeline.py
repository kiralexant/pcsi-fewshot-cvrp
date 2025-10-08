from __future__ import annotations

import copy
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import joblib
import numpy as np

import FewShotCVRP.bo.gp_fitting as gp_fitting
import FewShotCVRP.bo.kpcabo.kpcabo_torch as kpcabo_torch
import FewShotCVRP.examples.params_search.configs_loader as configs_loader
from FewShotCVRP.utils.logs import configure_logger

from .config import PathsConfig, TrainingConfig
from .interfaces import BatchEvaluator
from .simulation import CVRPFNNBatchEvaluator, make_cvrp_evaluator_factory
from .snapshotting import build_snapshotting_optimizer


class KernelPCANNTrainingPipeline:
    """Modular KPCA-BO training pipeline for neural parameter controllers."""

    def __init__(
        self,
        config: TrainingConfig,
        evaluator_factory=None,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.logger = logger or configure_logger("FewShotCVRPLogger")
        self._root_path = self._resolve_root_path(config.paths)
        self.evaluator_factory = (
            evaluator_factory
            if evaluator_factory is not None
            else make_cvrp_evaluator_factory(config.simulation)
        )

    @staticmethod
    def _resolve_root_path(paths: PathsConfig) -> Path:
        if paths.cur_root_path is not None:
            return paths.cur_root_path
        return Path(__file__).resolve().parents[2]

    def _export_config(self) -> Dict[str, Any]:
        cfg = self.config
        return {
            "random_seed": cfg.random_seed,
            "instances": list(cfg.instance_names),
            "simulation": cfg.simulation.as_mapping(),
            "parallel": {"num_procs": cfg.parallel.num_procs},
            "kpcabo": dict(cfg.kpcabo),
            "bo_embedding": dict(cfg.bo_embedding),
            "paths": {
                "cur_root_path": str(self._root_path),
                "strip_xml_extension": cfg.paths.strip_instance_extension,
                "results_subdir": cfg.paths.results_subdir,
                "precomputed_doe_filename": cfg.paths.precomputed_doe_filename,
            },
        }

    def _snapshotting_optimizer_class(self, experiment_subdir: str):
        return build_snapshotting_optimizer(
            kpcabo_torch,
            configs_loader,
            experiment_subdir=experiment_subdir,
            logger=self.logger,
        )

    def _maybe_load_precomputed_doe(self) -> Optional[Mapping[str, Any]]:
        bo_cfg = self.config.bo_embedding
        if str(bo_cfg.get("doe_method")) != "precomputed":
            return None
        path = (
            self._root_path
            / "examples"
            / "params_search"
            / self.config.paths.precomputed_doe_filename
        )
        if not path.exists():
            raise FileNotFoundError(f"Precomputed DoE file not found: {path}")
        self.logger.info("Loading precomputed DoE samples from %s", path)
        return joblib.load(path)

    def run(self) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        rng = np.random.default_rng(self.config.random_seed)
        precomputed = self._maybe_load_precomputed_doe()
        SnapshottingOptimizer = self._snapshotting_optimizer_class(
            self.config.paths.results_subdir
        )

        for instance_name in self.config.instance_names:
            self.logger.info("Starting KPCA-BO training for %s", instance_name)
            evaluator: BatchEvaluator = self.evaluator_factory(instance_name)
            lb, ub = evaluator.bounds
            dim = evaluator.dimension
            self.logger.debug("Search space dimension: %d", dim)

            kpcabo_cfg = copy.deepcopy(self.config.kpcabo)
            bo_cfg = copy.deepcopy(self.config.bo_embedding)
            mll_cfg_data = bo_cfg.get("mll_fit_config", {})
            mll_fit_cfg = gp_fitting.MLLFitConfig.model_validate(mll_cfg_data)

            for key in [
                "func",
                "f_batch",
                "bounds",
                "X_init",
                "y_init",
                "random_state",
                "cur_root_path",
                "bo_start_time",
                "instance_name",
                "experiment_config",
                "mll_fit_config",
                "bo_embedding_kwargs",
            ]:
                bo_cfg.pop(key, None)
                kpcabo_cfg.pop(key, None)

            if precomputed is not None:
                key = instance_name
                if key not in precomputed:
                    alt = Path(instance_name).stem
                    if alt in precomputed:
                        key = alt
                    else:
                        raise KeyError(
                            f"Instance {instance_name!r} missing in precomputed DoE file"
                        )
                arrays = precomputed[key]
                bo_cfg["X_init"] = arrays["X_"]
                bo_cfg["y_init"] = arrays["y_"]

            instance_seed = int(rng.integers(0, 10**9))
            bo_cfg["random_state"] = instance_seed
            bo_cfg["mll_fit_config"] = mll_fit_cfg

            num_procs = self.config.parallel.num_procs
            eval_rng = np.random.default_rng(instance_seed)

            def f_batch(nn_params: np.ndarray) -> np.ndarray:
                batch_seed = int(eval_rng.integers(0, 10**9))
                return evaluator.evaluate_batch(
                    nn_params,
                    rng_seed=batch_seed,
                    max_workers=num_procs,
                )

            kpcabo_cfg["bo_embedding_kwargs"] = bo_cfg

            run_stamp = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")
            instance_dir = (
                Path(instance_name).stem
                if self.config.paths.strip_instance_extension
                else instance_name
            )

            optimizer = SnapshottingOptimizer(
                func=None,
                f_batch=f_batch,
                bounds=list(zip(lb, ub)),
                random_state=instance_seed,
                cur_root_path=str(self._root_path),
                run_stamp=run_stamp,
                instance_name=instance_dir,
                experiment_config=self._export_config(),
                **kpcabo_cfg,
            )

            result = optimizer.run()
            optimizer.save_final_snapshot(result)
            results[instance_name] = result

            best_val = float(result.get("y_obs_best", float("inf")))
            self.logger.info(
                "Finished %s: best observed fitness %.3f", instance_name, best_val
            )

        return results


__all__ = ["KernelPCANNTrainingPipeline"]
