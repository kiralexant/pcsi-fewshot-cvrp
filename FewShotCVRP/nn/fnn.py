from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


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


class NNObjective:
    def __init__(
        self,
        theta_min: float = 1.0,
        theta_max: float = 100.0,
        in_dim=5,
        hidden_dims=(16, 16),
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        activation: Callable[[], nn.Module] = nn.SiLU,
    ):
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.net = BoundedMLP(in_dim, hidden_dims, theta_min, theta_max, activation).to(
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
        set_flat_params_(self.net, weights)

        f = self.build_function()

        L = float(simulate_and_get_L(f))
        return L


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


def nn_input_fitness(prv_fitness, cur_fitness):
    C = 10
    return np.exp(C * (np.log(cur_fitness) - np.log(prv_fitness) + 1)) / np.exp(C)


def nn_input_theta(theta):
    return (theta - 1) / 100
