from datetime import datetime
import os

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

import numpy as np

import cvrp_cpp as cc
import FewShotCVRP.dataset.parse_and_optimize as pao
import FewShotCVRP.ea as ea
import FewShotCVRP.examples.noise_study.noise_study_logger as nlg


def get_cvrp_instance(instance_name_str):
    dataset_dir = Path(pao.__file__).resolve().parent
    instance = pao.ProblemInstance(pao.ET.parse(dataset_dir / instance_name_str))
    return cc.CVRP(
        np.asarray(instance.depot_dist),
        np.asarray(instance.dist),
        np.asarray(instance.node_q),
        np.asarray(instance.capacity),
    )


def run_opt(
    theta_schedule,
    cvrp,
    seed,
    gen=50,
    lambda_=2 * 10**4,
    mutation=ea.mutate_shift_2opt_fast_fast,
):
    res = ea.one_plus_lambda_ea_with_theta_schedule(
        cvrp,
        seed=seed,
        lambda_=lambda_,
        max_evals=gen * lambda_,
        generations_number=gen,
        theta_schedule=theta_schedule,
        mutation=mutation,
        verbose=False,
    )
    return res


def _worker(instance_path: str, theta_schedule, seed: int, gen: int):
    cvrp = get_cvrp_instance(instance_path)  # свой инстанс на процесс
    return run_opt(theta_schedule, cvrp, seed, gen=gen)


def run_many_in_processes_nochunks(
    instance_path: str,
    theta_schedule,
    seeds,
    gen: int,
    num_procs: int | None = None,
):
    seeds = list(seeds)
    results = [None] * len(seeds)

    # На Linux лучше 'spawn', на Windows он и так по умолчанию
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass  # уже установлен — ок

    n_workers = num_procs or (os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        fut2idx = {
            ex.submit(_worker, instance_path, theta_schedule, s, gen): i
            for i, s in enumerate(seeds)
        }
        for fut in as_completed(fut2idx):
            i = fut2idx[fut]
            results[i] = fut.result()  # пробросит исключение, если было

    return results


if __name__ == "__main__":
    gen = 50
    file_path = Path(__file__).resolve().parent.parent
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d-%Hh%Mm%Ss")
    path = file_path / f"outputs/noise_study_{current_time}/noise_study_1.jsonl.gz"
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
        for theta_schedule in [
            np.ones(gen)
        ]:
            # theta_schedule = np.random.uniform(1.0, 100.0, gen)
            start = time.perf_counter()
            ress = run_many_in_processes_nochunks(
                instance_path,
                theta_schedule,
                seeds=range(1, 100),
                gen=gen,
                num_procs=100,  # число процессов
            )
            t = time.perf_counter() - start
            print(f"Done in {t:.2f} [s]")
            rec = nlg.make_record(
                [res["best_fitness"] for res in ress],
                theta_schedule=theta_schedule,
                instance_path=instance_path,
                gen=gen,
                lambda_=2 * 10**4,
                notes="Выставляем заранее заданное theta",
                extra={
                    "algo": "(1+λ) EA",
                    "lambda_policy": "preset",
                    "host": "liacs:viridium",
                },
            )
            nlg.append_record(str(path), rec)
            print("saved id:", rec["id"])
