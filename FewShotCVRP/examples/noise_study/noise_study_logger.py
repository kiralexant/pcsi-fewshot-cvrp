import datetime as dt
import json
import math
import uuid
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

import numpy as np


def _open_rw(path: str, mode: str):
    import gzip
    text_mode = 'b' not in mode
    if path.endswith(".gz"):
        if text_mode:
            return gzip.open(path, mode, encoding='utf-8')
        return gzip.open(path, mode)
    else:
        return open(path, mode, encoding='utf-8' if text_mode else None)

def _to_float_list(a: Iterable[float]) -> List[float]:
    return [
        (
            float(x)
            if (x is not None and not (isinstance(x, float) and math.isnan(x)))
            else None
        )
        for x in a
    ]


def make_record(
    values,
    *,
    instance_path: str,
    gen: int,
    lambda_: float,
    theta_schedule,
    seed: Optional[int] = None,
    notes: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    vals = np.asarray(list(values), dtype=float)
    theta = np.asarray(list(theta_schedule), dtype=float)

    if theta.size != int(gen):
        raise ValueError(f"theta_schedule size {theta.size} != gen {gen}")

    rec_id = str(uuid.uuid4())
    ts = dt.datetime.now().isoformat(timespec="seconds")

    stats = {
        "n": int(vals.size),
        "mean": float(vals.mean()) if vals.size else None,
        "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "min": float(vals.min()) if vals.size else None,
        "max": float(vals.max()) if vals.size else None,
    }

    record = {
        "id": rec_id,
        "ts": ts,
        "instance_path": instance_path,
        "gen": int(gen),
        "lambda": float(lambda_),
        "theta_schedule": _to_float_list(theta),  # <-- массив в JSON
        "seed": seed,
        "notes": notes,
        "extra": extra or {},
        "values": _to_float_list(vals),
        "stats": stats,
    }
    return record


def get_theta_by_id(path: str, rec_id: str) -> np.ndarray:
    for r in iter_records(path):
        if r.get("id") == rec_id:
            th = r.get("theta_schedule")
            if th is None:
                raise KeyError(f"id={rec_id} has no theta_schedule")
            return np.asarray(th, dtype=float)
    raise KeyError(f"id={rec_id} not found")


def append_record(path: str, record: Dict[str, Any]) -> None:
    # одна JSON-строка на запись (NDJSON)
    with _open_rw(path, "at") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_records(path: str) -> Iterator[Dict[str, Any]]:
    with _open_rw(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def query(path: str, pred: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
    return [r for r in iter_records(path) if pred(r)]


def get_values_by_id(path: str, rec_id: str) -> np.ndarray:
    for r in iter_records(path):
        if r.get("id") == rec_id:
            return np.asarray(r["values"], dtype=float)
    raise KeyError(f"id={rec_id} not found")
