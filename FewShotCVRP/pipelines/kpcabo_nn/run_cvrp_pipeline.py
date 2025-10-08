from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_training_config
from .pipeline import KernelPCANNTrainingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Kernel-PCA-BO training for CVRP theta-control neural networks.",
    )
    default_config = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "params_search"
        / "kpca-bo-nn-config.json"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"Path to training config JSON (default: {default_config})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    pipeline = KernelPCANNTrainingPipeline(config)
    results = pipeline.run()
    summary = {
        name: {
            "y_obs_best": float(res.get("y_obs_best", float("inf"))),
            "evaluations": int(res.get("n_evals", 0)),
        }
        for name, res in results.items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
