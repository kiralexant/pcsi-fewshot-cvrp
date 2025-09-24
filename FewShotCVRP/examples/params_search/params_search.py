from datetime import datetime
import os

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
    return res["best_fitness"]


def _worker(instance_path: str, theta_schedule, seed: int, gen: int):
    cvrp = get_cvrp_instance(instance_path)
    return run_opt(theta_schedule, cvrp=cvrp, seed=seed, gen=gen)


def run_opt_batch(
    theta_schedules_list: np.ndarray,
    instance_path: str,
    gen: int,
    seed: int,
    num_procs: int | None = None,
):
    results = np.ones(len(theta_schedules_list)) * np.inf

    # На Linux лучше 'spawn', на Windows он и так по умолчанию
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass  # уже установлен — ок

    n_workers = num_procs or (os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        fut2idx = {
            ex.submit(_worker, instance_path, theta_schedule, seed, gen): i
            for i, theta_schedule in enumerate(theta_schedules_list)
        }
        for fut in as_completed(fut2idx):
            i = fut2idx[fut]
            results[i] = fut.result()  # пробросит исключение, если было

    return results


if __name__ == "__main__":
    seed = 2
    instance_path = "X-n101-k25.xml"
    rng = np.random.default_rng(seed)
    num_procs = 100
    gen = 5
    f = None
    f_batch = lambda theta_schedule_list: run_opt_batch(
        theta_schedules_list=theta_schedule_list,
        instance_path=instance_path,
        gen=gen,
        seed=rng.integers(1, 10**9),
        num_procs=num_procs,
    )
    bo = bo_pure.BayesianOptimizer(
        f,
        f_batch=f_batch,
        bounds=[(1.0, 100.0)] * gen,
        n_init=100,
        n_iter=20,
        sigma=600,
        noise_std_guess=600,
        noise_std_bounds=(10, 1000),
        kernel="rbf",
        length_scale_bounds=(1e-2, 1e7),
        c_factor_bounds=(1 / 2, 2),
        suggestions_per_step=100,  # number of the diverse local maxima of EI per iteration
        diversity_frac=0.1,  # 10% of average box size as min separation
        random_state=seed,
    )
    result = bo.run()
    gp = bo.get_gp()
    print("Best x:", result["x_best"], "Best y:", result["y_best"])
    print("Kernel:", gp.kernel_)
    print("\nARD report:")
    bo.report_ard()
    best_perm, best_v = bo.recommend()
    print("Recommended x:", best_perm, "Recommended y:", best_v)
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d-%Hh%Mm%Ss")
    examples_path = Path(__file__).resolve().parent.parent
    bo.save_snapshot(examples_path / f"outputs/bo-static-{current_time}")
