import importlib
import json
from pathlib import Path


def _resolve_dotted_attr(path: str):
    """
    Import and return an attribute by its fully-qualified dotted path.
    E.g. 'torch.nn.SiLU' or 'ea.mutate_shift_2opt_fast_fast'.
    """
    if not isinstance(path, str) or "." not in path:
        raise ValueError(f"Expected dotted path string, got: {path!r}")
    module_name, _, attr = path.rpartition(".")
    mod = importlib.import_module(module_name)
    return getattr(mod, attr)


def function_to_dotted_attr(cfg):
    sim = cfg.get("simulation", {})
    if "ea_mutation_operator" in sim and callable(sim["ea_mutation_operator"]):
        k = sim["ea_mutation_operator"]
        sim["ea_mutation_operator"] = f"{k.__module__}.{k.__name__}"
    if "activation" in sim and callable(sim["activation"]):
        k = sim["activation"]
        sim["activation"] = f"{k.__module__}.{k.__name__}"


def load_experiment_config(config_path: Path | None = None) -> dict:
    """
    Load JSON config and resolve any dotted callables/classes.
    Assumes file is next to this script if path not provided.
    """
    if config_path is None:
        config_path = Path(__file__).with_name("nn-experiment-config.json")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Resolve callables/classes
    sim = cfg.get("simulation", {})
    if "ea_mutation_operator" in sim and isinstance(sim["ea_mutation_operator"], str):
        sim["ea_mutation_operator"] = _resolve_dotted_attr(sim["ea_mutation_operator"])
    if "activation" in sim and isinstance(sim["activation"], str):
        sim["activation"] = _resolve_dotted_attr(sim["activation"])

    cfg["simulation"] = sim
    return cfg
