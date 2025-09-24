# pip install torch
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from FewShotCVRP.bo import bo_pure

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")


import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cvrp_cpp as cc
import numpy as np

import FewShotCVRP.dataset.parse_and_optimize as pao
import FewShotCVRP.ea as ea
import FewShotCVRP.examples.noise_study.noise_study_logger as nlg


@dataclass
class SimulationConstants:
    cvrp_instance_str: str
    random_seed: int

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


# ---------- 1) Сеть: 5 -> hidden -> hidden -> 1, выход в [1, 100] ----------
class BoundedMLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 5,
        hidden_dims: Iterable[int] = (16, 16),
        theta_min: float = 1.0,
        theta_max: float = 100.0,
        activation: Callable[[], nn.Module] = nn.SiLU,  # мягкая и устойчивая
    ):
        super().__init__()
        assert theta_max > theta_min
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)

        layers = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), activation()]
            last = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(last, 1)  # без активации тут

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., 5) тензор. Возвращает (...,) в диапазоне [theta_min, theta_max].
        """
        z = self.head(self.body(x))
        y01 = torch.sigmoid(z)  # в [0, 1]
        y = self.theta_min + (self.theta_max - self.theta_min) * y01
        return y.squeeze(-1)


# ---------- 2) Утилиты для работы с плоским вектором параметров ----------
def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_flat_params(model: nn.Module) -> torch.Tensor:
    return parameters_to_vector([p.detach() for p in model.parameters()])


def set_flat_params_(model: nn.Module, flat: Union[torch.Tensor, np.ndarray]) -> None:
    if isinstance(flat, np.ndarray):
        flat = torch.from_numpy(flat)
    flat = flat.to(
        next(model.parameters()).device, dtype=next(model.parameters()).dtype
    )
    vector_to_parameters(flat, list(model.parameters()))


# ---------- 3) Обёртка: веса -> функция f(x) для вашей симуляции ----------
@dataclass
class FunctionHandle:
    model: nn.Module
    device: torch.device
    dtype: torch.dtype

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[float, np.ndarray]:
        """
        Принимает x формы (5,) или (N,5); возвращает float или np.ndarray с диапазоном [1, 100].
        """
        self.model.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                t = torch.from_numpy(x).to(self.device, self.dtype)
            else:
                t = x.to(self.device, self.dtype)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            y = self.model(t)  # shape: (N,)
            y_np = y.cpu().numpy()
            return float(y_np[0]) if y_np.shape[0] == 1 else y_np


# ---------- 4) Пример сопряжения с внешним оптимизатором ----------
# Предположим, у вас есть:
#   - simulate_and_get_L(f: Callable[[np.ndarray], float]) -> float
#   - внешний оптимизатор, который работает с непрерывным вектором весов
#
# Ниже — "объектив" для оптимизатора: на вход вектор весов, на выход L.


class NNObjective:
    def __init__(
        self,
        theta_min: float = 1.0,
        theta_max: float = 100.0,
        in_dim=5,
        hidden_dims=(16, 16),
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.net = BoundedMLP(in_dim, hidden_dims, theta_min, theta_max).to(
            self.device, self.dtype
        )

        # Заодно подготовим «нулевой» вектор весов для инициализации оптимизатора
        self.w0 = get_flat_params(self.net).cpu().numpy()

    def build_function(self) -> FunctionHandle:
        return FunctionHandle(model=self.net, device=self.device, dtype=self.dtype)

    def __call__(
        self,
        weights: Union[np.ndarray, torch.Tensor],
        simulate_and_get_L: Callable[[Callable], float],
    ) -> float:
        # 1) применяем веса
        set_flat_params_(self.net, weights)

        # 2) строим вызываемый объект f(x)
        f = self.build_function()

        # 3) считаем качество в вашей симуляции
        L = float(simulate_and_get_L(f))
        return L


def get_cvrp_instance(instance_name_str):
    dataset_dir = Path(pao.__file__).resolve().parent
    instance = pao.ProblemInstance(pao.ET.parse(dataset_dir / instance_name_str))
    return cc.CVRP(
        np.asarray(instance.depot_dist),
        np.asarray(instance.dist),
        np.asarray(instance.node_q),
        np.asarray(instance.capacity),
    )


def _worker(nn_weights: np.ndarray, constants: SimulationConstants, seed: int):
    cvrp = get_cvrp_instance(constants.cvrp_instance_str)
    objective = NNObjective(
        theta_min=constants.theta_min,
        theta_max=constants.theta_max,
        in_dim=constants.in_dim,
        hidden_dims=constants.hidden_dims,
    )

    simulation = lambda predict_theta: ea.one_plus_lambda_ea_with_theta_control(
        cvrp,
        theta_schedule_window=[100.0] * constants.ea_window_size,
        window=constants.ea_window_size,
        theta_control_fun=predict_theta,
        seed=seed,
        lambda_=constants.ea_lambda_,
        max_evals=constants.ea_generations_number * constants.ea_lambda_,
        generations_number=constants.ea_generations_number,
        mutation=ea.mutate_shift_2opt_fast_fast,
        verbose=False,
    )["best_fitness"]
    return objective(nn_weights, simulation)


def run_batches(
    seeds: np.ndarray,
    nn_weights: np.ndarray,
    constants: SimulationConstants,
    num_procs: int | None = None,
):
    results = np.ones(len(seeds)) * np.inf

    # На Linux лучше 'spawn', на Windows он и так по умолчанию
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass  # уже установлен — ок

    n_workers = num_procs or (os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        fut2idx = {
            ex.submit(_worker, nn_weights, constants, seed): i
            for i, seed in enumerate(seeds)
        }
        for fut in as_completed(fut2idx):
            i = fut2idx[fut]
            results[i] = fut.result()  # пробросит исключение, если было

    return results


import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector


def n_params(model: nn.Module) -> int:
    return parameters_to_vector(model.parameters()).numel()  # размерность вектора весов


def make_bounds_for_linear_mlp(
    model: nn.Module,
    scheme: str = "kaiming",  # "kaiming" | "xavier"
    alpha_w: float = 2.0,  # расширение границ для весов
    alpha_b: float = 2.0,  # расширение границ для bias
) -> tuple[np.ndarray, np.ndarray]:
    """
    Возвращает два массива нижних/верхних границ (lb, ub) длины n_params(model),
    выстроенных в том же порядке, что и parameters_to_vector(model.parameters()).
    """

    lbs = []
    ubs = []

    for m in model.modules():
        if isinstance(m, nn.Linear):
            # --- веса ---
            W = m.weight  # shape: (out_features, in_features)
            fan_out, fan_in = W.shape

            if scheme == "xavier":
                # a = sqrt(6 / (fan_in + fan_out))  (Glorot/Xavier uniform)
                a = np.sqrt(6.0 / (fan_in + fan_out))
                bw = alpha_w * a
            else:
                # He/Kaiming uniform для ReLU-подобных (SiLU≈ReLU по масштабу)
                # bound = sqrt(3) * gain * sqrt(2 / fan_in)   (из kaiming_uniform_)
                gain = float(nn.init.calculate_gain("relu"))  # ~sqrt(2)
                bw = alpha_w * np.sqrt(3.0) * gain * np.sqrt(1.0 / fan_in)

            lbs.append(np.full(W.numel(), -bw, dtype=np.float32))
            ubs.append(np.full(W.numel(), bw, dtype=np.float32))

            # --- bias (если есть) ---
            if m.bias is not None:
                # По умолчанию в PyTorch bias ~ U(-1/sqrt(fan_in), +1/sqrt(fan_in))
                bb = alpha_b * (1.0 / np.sqrt(fan_in))
                bsz = m.bias.numel()
                lbs.append(np.full(bsz, -bb, dtype=np.float32))
                ubs.append(np.full(bsz, bb, dtype=np.float32))

    lb = np.concatenate(lbs)
    ub = np.concatenate(ubs)

    # sanity-check: совпасть с длинной вектора параметров
    assert lb.size == n_params(model) == ub.size
    return lb, ub


# ---------- 5) Пример использования с псевдо-оптимизатором ----------
if __name__ == "__main__":

    constants = SimulationConstants(
        cvrp_instance_str="X-n101-k25.xml",
        random_seed=None,
        ea_window_size=5,
        ea_generations_number=50,
        ea_lambda_=2 * 10**4,
        ea_mutation_operator=ea.mutate_shift_2opt_fast_fast,
        in_dim=10,
        hidden_dims=[15],
        theta_min=1.0,
        theta_max=100.0,
        activation=nn.SiLU,
    )

    from FewShotCVRP.bo.bo_pure import BayesianOptimizer

    file_path = Path(__file__).resolve().parent.parent

    snap = BayesianOptimizer.load_snapshot(
        file_path / "outputs/bo-static-2025-08-22-05h06m04s"
    )

    objective = NNObjective(
        theta_min=constants.theta_min,
        theta_max=constants.theta_max,
        in_dim=constants.in_dim,
        hidden_dims=constants.hidden_dims,
    )
    nn_weights = np.array(snap.result["x_best"])

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d-%Hh%Mm%Ss")
    path = file_path / f"outputs/noise_study_nn_{current_time}/noise_study_1.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)

    for instance_path in [
        "X-n101-k25.xml",
        # "X-n153-k22.xml",
        # "X-n176-k26.xml",
        # "X-n491-k59.xml",
        # "X-n308-k13.xml",
        # "X-n209-k16.xml",
        # "X-n237-k14.xml",
        # "X-n280-k17.xml",
        # "X-n420-k130.xml",
        # "X-n524-k137.xml",
    ]:
        seeds = np.arange(1, 101)
        start = time.perf_counter()
        ress = run_batches(
            seeds,
            nn_weights,
            constants,
            num_procs=100,  # число процессов
        )
        t = time.perf_counter() - start
        print(f"Done in {t:.2f} [s]")
        rec = nlg.make_record(
            ress,
            theta_schedule=[100.0] * constants.ea_generations_number,
            instance_path=instance_path,
            gen=constants.ea_generations_number,
            lambda_=constants.ea_lambda_,
            notes="theta контролируется нейронной сетью, нет фиксированного расписания",
            extra={
                "algo": "(1+λ) EA",
                "theta_policy": "nn",
                "lambda_policy": "preset",
                "host": "liacs:viridium",
            },
        )
        nlg.append_record(str(path), rec)
        print("saved id:", rec["id"])
