import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
from botorch.utils.sampling import draw_sobol_samples
from numpy.typing import ArrayLike
from scipy.stats import qmc

import FewShotCVRP.bo.bo_torch as bo_torch
import FewShotCVRP.examples.params_search.configs_loader as config_loader
import FewShotCVRP.examples.params_search.simulation as simulation
import FewShotCVRP.nn.fnn as fnn


@dataclass
class RSSaveSnapshots:
    func: Callable[[np.ndarray], float] = None
    f_batch: Optional[Callable[[np.ndarray], ArrayLike]] = None

    n_iter: int = 2000
    random_state: int = 1
    doe_method: str = "sobol"
    minimize: bool = True

    bounds_np: Optional[np.ndarray] = None
    cur_root_path: str = ""
    bo_start_time: str = ""
    cvrp_instance_str: str = ""

    # Device / dtype
    device: Union[str, torch.device] = "cpu"
    dtype: torch.dtype = torch.double

    def __post_init__(self):
        lb_t = torch.as_tensor(
            self.bounds_np[:, 0], dtype=self.dtype, device=self.device
        )
        ub_t = torch.as_tensor(
            self.bounds_np[:, 1], dtype=self.dtype, device=self.device
        )
        self.bounds_t = torch.stack([lb_t, ub_t])  # [2, d]
        self.dim = self.bounds_np.shape[0]

    def run(self) -> None:
        X = self._sample_doe(self.n_iter)
        y = self._evaluate_objective_np(X)
        self.X_ = X
        self.y_ = y

    def _evaluate_objective_np(self, X_np: np.ndarray) -> np.ndarray:
        """Evaluate objective(s) given numpy float64 array (n,d) -> returns numpy (n,)."""
        if self.f_batch is not None:
            y = np.asarray(
                self.f_batch(np.asarray(X_np, dtype=np.float64)), dtype=np.float64
            ).reshape(-1)
            assert (
                y.shape[0] == X_np.shape[0]
            ), "f_batch must return n values for n inputs"
            return y
        # scalar objective
        ys = [
            float(self.func(np.asarray(x, dtype=np.float64).reshape(-1))) for x in X_np
        ]
        return np.asarray(ys, dtype=np.float64)

    def _sample_doe(self, n: int) -> np.ndarray:
        """Return n samples in the box (numpy float64)."""
        d = self.dim
        m = self.doe_method.lower()
        if m == "sobol":
            bounds_t = self.bounds_t
            X = draw_sobol_samples(
                bounds=bounds_t, n=n, q=1, seed=self.random_state
            ).squeeze(-2)
            return bo_torch._to_numpy64(X)  # (n,d)
        if m == "random":
            lb = self.bounds_np[:, 0]
            ub = self.bounds_np[:, 1]
            U = self.rng_.random((n, d), dtype=np.float64)
            return lb + (ub - lb) * U
        if m == "lhs":
            sampler = qmc.LatinHypercube(d=d, seed=self.random_state)
            U = sampler.random(n=n)
            return qmc.scale(U, self.bounds_np[:, 0], self.bounds_np[:, 1]).astype(
                np.float64
            )
        raise ValueError(f"Unknown doe_method={self.doe_method!r}")

    # ---------------- snapshot ----------------
    def save_snapshot(self, save_dir: str) -> Path:
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)

        arrays = {
            "X_": self.X_,
            "y_": self.y_,
            "bounds": np.asarray(self.bounds_np, dtype=np.float64),
        }
        np.savez_compressed(d / "arrays.npz", **arrays)

        manifest = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "numpy_version": np.__version__,
        }
        (d / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        y = arrays["y_"]
        X = arrays["X_"]
        if y.size:
            idx = int(np.argmin(y) if self.minimize else np.argmax(y))
            x_best = X[idx].tolist()
            y_best = float(y[idx])
        else:
            x_best, y_best = None, None
        result_json = {
            "x_obs_best": x_best,
            "y_obs_best": y_best,
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
        }

        (d / "result.json").write_text(
            json.dumps(result_json, indent=2), encoding="utf-8"
        )
        return d

    @staticmethod
    def load_snapshot(load_dir: str):
        d = Path(load_dir)
        arrays_file = np.load(d / "arrays.npz")
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        return arrays_file, manifest

    def save_final_snapshot(self):
        self.save_snapshot(
            self.cur_root_path
            / f"outputs"
            / f"{self.bo_start_time}"
            / f"per-instance-param-control"
            / f"{self.cvrp_instance_str}"
            / f"rs-final"
        )


if __name__ == "__main__":

    cfg = config_loader.load_experiment_config(
        Path(__file__).with_name("rs-experiment-config.json")
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
        bo_cfg = cfg["rs"]
        rs = RSSaveSnapshots(
            None,
            f_batch=f_batch,
            n_iter=int(bo_cfg["n_iter"]),
            bounds_np=np.asarray(list(zip(lb, ub)), dtype=np.float64),
            doe_method=str(bo_cfg["doe_method"]),
            random_state=random_seed,
            cur_root_path=cur_root_path,
            bo_start_time=datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss"),
            cvrp_instance_str=(
                os.path.splitext(cvrp_instance_str)[0]
                if strip_xml_ext
                else cvrp_instance_str
            ),
        )

        rs.run()
        rs.save_final_snapshot()
