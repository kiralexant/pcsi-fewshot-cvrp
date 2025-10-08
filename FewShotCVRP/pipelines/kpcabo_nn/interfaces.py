from __future__ import annotations

from typing import Callable, Protocol, Sequence

import numpy as np


class BatchEvaluator(Protocol):
    """Protocol for batched black-box evaluations of controller weights."""

    @property
    def dimension(self) -> int:
        ...

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        ...

    def evaluate_batch(
        self,
        candidates: Sequence[Sequence[float]] | np.ndarray,
        rng_seed: int,
        max_workers: int | None = None,
    ) -> np.ndarray:
        ...


EvaluatorFactory = Callable[[str], BatchEvaluator]


__all__ = ["BatchEvaluator", "EvaluatorFactory"]
