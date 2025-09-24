import json
import os

import joblib

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import FewShotCVRP.bo.bo_torch as bo_torch
import FewShotCVRP.bo.gp_fitting as gp_fitting
import FewShotCVRP.examples.params_search.configs_loader as config_loader
import FewShotCVRP.examples.params_search.simulation as simulation
import FewShotCVRP.nn.fnn as fnn


@dataclass
class BOSaveSnapshots(bo_torch.BayesianOptimizer):
    cur_root_path: str = ""
    bo_start_time: str = ""
    cvrp_instance_str: str = ""
    experiment_config: Dict = None

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

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        super().step()
        self.save_snapshot(
            self.cur_root_path
            / f"outputs"
            / f"{self.bo_start_time}"
            / f"per-instance-param-control"
            / f"{self.cvrp_instance_str}"
            / f"bo-{self.bo_iteration_number}"
        )
        best_so_far_performance = min(self.y_.squeeze(-1))
        self.logger.log(logging.INFO, f"Best-so-far: {int(best_so_far_performance)}")
        self.bo_iteration_number += 1

    def save_final_snapshot(self, results):
        self.save_snapshot(
            self.cur_root_path
            / f"outputs"
            / f"{self.bo_start_time}"
            / f"per-instance-param-control"
            / f"{self.cvrp_instance_str}"
            / f"bo-final",
            results=results,
        )


if __name__ == "__main__":

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
    if str(bo_cfg["doe_method"]) == "precomputed":
        precomputed = joblib.load(
            cur_root_path / "params_search" / "precomputed_DoEs.joblib"
        )

    # Loop over CVRP instances from config
    for cvrp_instance_str in cfg["cvrp_instances"]:
        simulation_rng = np.random.default_rng(random_seed)

        # --- Simulation constants from config ---
        sim = cfg["simulation"]
        constants = simulation.SimulationConstants(
            cvrp_instance_str=cvrp_instance_str,
            ea_window_size=int(sim["ea_window_size"]),
            ea_generations_number=int(sim["ea_generations_number"]),
            ea_lambda_=int(sim["ea_lambda"]),
            ea_mutation_operator=sim["ea_mutation_operator"],  # resolved callable
            in_dim=int(sim["in_dim"]),
            hidden_dims=list(sim["hidden_dims"]),
            theta_min=float(sim["theta_min"]),
            theta_max=float(sim["theta_max"]),
            activation=sim["activation"],  # resolved class
        )

        # --- Objective and bounds (unchanged logic) ---
        objective = fnn.NNObjective(
            theta_min=constants.theta_min,
            theta_max=constants.theta_max,
            in_dim=constants.in_dim,
            hidden_dims=constants.hidden_dims,
        )
        dim = fnn.n_params(objective.net)
        lb, ub = fnn.make_bounds_for_linear_mlp(objective.net)

        # --- Batch runner with configurable processes ---
        num_procs = int(cfg.get("parallel", {}).get("num_procs", 1))
        f_batch = lambda nn_params: simulation.run_parallel_fnn(
            nn_weights_array=nn_params,
            constants=constants,
            simulation_rng=simulation_rng,
            num_procs=num_procs,
        )

        # --- BO parameters from config ---
        bo_cfg = cfg["bo"]

        # --- Load precomputed DoE ---
        X_init, y_init = None, None
        if str(bo_cfg["doe_method"]) == "precomputed":
            arrays = precomputed[cvrp_instance_str]
            X_init = arrays["X_"]
            y_init = arrays["y_"]

        # --- BO kwargs from config, but drop things we set in code ---
        bo_cfg = dict(cfg["bo"])  # shallow copy
        mll_fit_config_dict = bo_cfg["mll_fit_config"]
        mll_fit_config = gp_fitting.MLLFitConfig.model_validate(mll_fit_config_dict)
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

        # --- Create BO (your updated signature) ---
        bo = BOSaveSnapshots(
            None,
            f_batch=f_batch,
            bounds=list(zip(lb, ub)),
            X_init=X_init,
            y_init=y_init,
            random_state=random_seed,
            cur_root_path=cur_root_path,
            bo_start_time=datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss"),
            cvrp_instance_str=(
                os.path.splitext(cvrp_instance_str)[0]
                if strip_xml_ext
                else cvrp_instance_str
            ),
            experiment_config=cfg,
            mll_fit_config=mll_fit_config,
            **bo_cfg,  # all remaining BO hyperparameters from JSON
        )

        result = bo.run()
        gp = bo.get_gp()
        print("Best x:", result["x_obs_best"], "Best y:", result["y_obs_best"])
        print("\nARD report:")
        bo.report_ard()
        bo.save_final_snapshot(result)
