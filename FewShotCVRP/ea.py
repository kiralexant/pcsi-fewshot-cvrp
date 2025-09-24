from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numba import njit

import cvrp_cpp as cc
import FewShotCVRP.dataset.parse_and_optimize as pao
import FewShotCVRP.utils.fixed_queue as fq

# type of mutation: takes current permutation and numpy RNG -> returns new permutation
MutationFn = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def is_permutation(p: Sequence[int], n: int) -> bool:
    if len(p) != n:
        return False
    return set(p) == set(range(n))


def default_swap_mutation(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Swap two random positions."""
    q = p.copy()
    i, j = rng.integers(0, len(p), size=2)
    while j == i:
        j = rng.integers(0, len(p))
    q[i], q[j] = q[j], q[i]
    return q


# Mutation: a 2-opt elementary mutation which chooses two random indices and
# reverses the permutation between them
def mutate_once_2opt(p: np.ndarray, rng: np.random.Generator):
    q = np.array(p, copy=True)
    i1 = rng.integers(0, len(q))
    i2 = rng.integers(0, len(q))
    if i1 > i2:
        i1, i2 = i2, i1
    while i1 < i2:
        q[i1], q[i2] = q[i2], q[i1]
        i1 += 1
        i2 -= 1
    return q


# Mutation: a swap-element mutation which chooses two random indices and
# exchanges the elements at these indices.
# Some people use 'swap' for swapping adjacent elements. English is ambiguous, so we don't do it.
def mutate_once_swap(p: np.ndarray, rng: np.random.Generator):
    q = np.array(p, copy=True)
    i1 = rng.integers(0, len(q))
    i2 = rng.integers(0, len(q))
    q[i1], q[i2] = q[i2], q[i1]
    return q


def mutate_shift(
    p: np.ndarray, rng: np.random.Generator, theta: float, mutate_once: MutationFn
):
    k = rng.poisson(theta)
    # shift if mutation strength is 0
    if k == 0:
        k += 1
    for _ in range(k):
        p = mutate_once(p, rng)
    return p


def mutate_shift_swap(p: np.ndarray, rng: np.random.Generator, theta: float):
    q = np.array(p, copy=True)
    nq = len(q)
    k = rng.poisson(theta)
    # shift if mutation strength is 0
    if k == 0:
        k += 1
    for _ in range(k):
        i1 = rng.integers(0, nq)
        i2 = rng.integers(0, nq)
        q[i1], q[i2] = q[i2], q[i1]
    return q


def mutate_shift_swap_fast(p: np.ndarray, rng: np.random.Generator, theta: float):
    # копирование быстрее через .copy() чем через np.array(..., copy=True)
    q = p.copy()
    n = q.shape[0]

    k = rng.poisson(theta)
    if k <= 0:
        k = 1

    # сгенерировали все пары сразу (меньше overhead на Python уровне)
    pairs = rng.integers(0, n, size=(k, 2), dtype=np.int64)
    i1 = pairs[:, 0]
    i2 = pairs[:, 1]

    # локальная ссылка на массив (микро-оптимизация)
    qq = q
    # последовательные свопы — эквивалентны исходной логике
    for a, b in zip(i1, i2):
        qq[a], qq[b] = qq[b], qq[a]

    return qq


@njit(cache=True, nogil=True)
def _apply_swaps_inplace(q, i1, i2):
    k = i1.size
    for t in range(k):
        a = i1[t]
        b = i2[t]
        if a != b:
            tmp = q[a]
            q[a] = q[b]
            q[b] = tmp


def mutate_shift_swap_fast_fast(p: np.ndarray, rng: np.random.Generator, theta: float):
    q = np.array(p, copy=True)
    k = rng.poisson(theta)
    if k <= 0:
        k = 1
    pairs = rng.integers(0, q.shape[0], size=(k, 2), dtype=np.int64)
    _apply_swaps_inplace(q, pairs[:, 0], pairs[:, 1])
    return q


def mutate_shift_2opt(p: np.ndarray, rng: np.random.Generator, theta: float):
    q = np.array(p, copy=True)
    nq = len(q)
    k = rng.poisson(theta)
    # shift if mutation strength is 0
    if k == 0:
        k += 1
    for _ in range(k):
        i1 = rng.integers(0, nq)
        i2 = rng.integers(0, nq)
        if i1 > i2:
            i1, i2 = i2, i1
        while i1 < i2:
            q[i1], q[i2] = q[i2], q[i1]
            i1 += 1
            i2 -= 1
    return q


@njit(cache=True, nogil=True)
def _apply_2opt_inplace(q, i1, i2):
    k = i1.size
    for t in range(k):
        a = i1[t]
        b = i2[t]
        if a > b:
            tmp = a
            a = b
            b = tmp
        while a < b:
            tmp = q[a]
            q[a] = q[b]
            q[b] = tmp
            a += 1
            b -= 1


def mutate_shift_2opt_fast_fast(
    p: np.ndarray, rng: np.random.Generator, theta: float
) -> np.ndarray:
    q = np.array(p, copy=True)
    k = rng.poisson(theta)
    if k <= 0:
        k = 1
    pairs = rng.integers(0, q.shape[0], size=(k, 2), dtype=np.int64)
    _apply_2opt_inplace(q, pairs[:, 0], pairs[:, 1])
    return q


def fitness(cvrp, p):
    return cvrp.dynamic_programming_fitness(p)


def one_plus_lambda_ea(
    cvrp,  # instance of cvrp_cpp.CVRP
    theta: float = 1.0,
    lambda_: int = 100,  # offspring per generation
    max_evals: int = 100_000,  # total objective evaluations budget
    mutation: Optional[MutationFn] = None,  # generic mutation callback
    init_perm: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
    patience: Optional[int] = None,  # stop after this many generations w/o improvement
    return_routes: bool = False,  # if True, also return DP routes for the best perm
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Minimizes cvrp.dynamic_programming_fitness over permutations using (1+λ)-EA.
    Returns dict with best solution and log.
    """
    rng = np.random.default_rng(seed)
    n = int(cvrp.n)

    if mutation is None:
        mut = lambda p, rng_: mutate_shift_swap(p, rng_, theta)
    else:
        mut = lambda p, rng_: mutation(p, rng_, theta)

    # initialize parent
    if init_perm is None:
        parent = np.arange(n, dtype=int)
        rng.shuffle(parent)
    else:
        parent = np.array(init_perm, dtype=int)
        assert is_permutation(parent, n), "init_perm must be a permutation of 0..n-1"

    # evaluate parent
    best_val = fitness(cvrp, parent)
    evals = 1
    history: List[Tuple[int, float, float]] = [(evals, -1.0, best_val)]
    gens = 0
    no_improv_gens = 0

    if verbose:
        print(f"[init] fitness={best_val:.6f}")

    while evals < max_evals:
        gens += 1
        best_child_val = None
        best_child = None

        # generate λ offspring
        for _ in range(lambda_):
            child = mut(parent, rng)

            # early cutoff using best so far
            child_val = fitness(cvrp, child)
            evals += 1

            if (best_child_val is None) or (child_val < best_child_val):
                best_child_val = child_val
                best_child = child

            if evals >= max_evals:
                break

        # elitist selection (ties keep parent)
        improved = best_child_val is not None and best_child_val < best_val
        if improved:
            parent = best_child
            best_val = best_child_val
            no_improv_gens = 0
        else:
            no_improv_gens += 1

        history.append((evals, theta, best_val))
        if verbose:
            tag = "↑" if improved else "·"
            print(f"[gen {gens}] fitness={best_val:.6f} {tag} evals={evals}")

        if patience is not None and no_improv_gens >= patience:
            if verbose:
                print(
                    f"[stop] patience reached: {patience} generations without improvement"
                )
            break

    result: Dict[str, Any] = {
        "best_perm": parent,
        "best_fitness": best_val,
        "evals": evals,
        "gens": gens,
        "history": history,
    }

    if return_routes:
        # use exact evaluation without cutoff to get routes (and to assert consistency)
        cost, routes = fitness(cvrp, parent)
        result["routes"] = routes  # list[list[int]], each row is a route (no -1)
        result["best_fitness_exact"] = float(cost)

    return result


def one_plus_lambda_ea_with_theta_schedule(
    cvrp,  # instance of cvrp_cpp.CVRP
    lambda_: int = 100,  # offspring per generation
    max_evals: int = 100_000,  # total objective evaluations budget
    mutation: Optional[MutationFn] = None,
    init_perm: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
    patience: Optional[int] = None,  # stop after this many generations w/o improvement
    return_routes: bool = False,  # if True, also return DP routes for the best perm
    verbose: bool = False,
    generations_number: Optional[int] = None,
    theta_schedule: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    (1+λ)-EA with a pre-set theta per generation.

    For generation g (0-based), uses theta = theta_schedule[g] inside mutate_shift.
    Effective evaluation budget is adjusted to: min(max_evals, lambda_ * generations_number).
    """
    if generations_number is None or theta_schedule is None:
        raise ValueError("Provide both generations_number and theta_schedule.")
    if len(theta_schedule) != generations_number:
        raise ValueError("theta_schedule length must equal generations_number.")
    if lambda_ <= 0:
        raise ValueError("lambda_ must be positive.")

    rng = np.random.default_rng(seed)
    n = int(cvrp.n)

    # initialize parent
    if init_perm is None:
        parent = np.arange(n, dtype=int)
        rng.shuffle(parent)
    else:
        parent = np.array(init_perm, dtype=int)
        assert is_permutation(parent, n), "init_perm must be a permutation of 0..n-1"

    # evaluate parent (total eval counter includes this one)
    best_val = fitness(cvrp, parent)
    evals_total = 1
    evals_children = 0
    child_budget = int(min(max_evals, lambda_ * generations_number))

    history: List[Tuple[int, float, float]] = [(evals_total, -1.0, float(best_val))]
    gens_done = 0
    no_improv_gens = 0

    if verbose:
        print(f"[init] fitness={best_val:.6f}")

    # main loop over generations with scheduled theta
    for g in range(generations_number):
        if evals_children >= child_budget:
            break
        gens_done += 1

        theta = float(theta_schedule[g])

        best_child_val = None
        best_child = None

        if mutation is None:
            mut = lambda p: mutate_shift_swap_fast_fast(p, rng, theta)
        else:
            mut = lambda p: mutation(p, rng, theta)

        # generate λ offspring for this generation (respect eval budget)
        for _ in range(lambda_):
            if evals_children >= child_budget:
                break
            # mutate with current generation's theta

            child = mut(parent)

            child_val = fitness(cvrp, child)
            evals_children += 1
            evals_total += 1

            if (best_child_val is None) or (child_val < best_child_val):
                best_child_val = child_val
                best_child = child

        # elitist selection (ties keep parent)
        improved = (best_child_val is not None) and (best_child_val < best_val)
        if improved:
            parent = best_child
            best_val = best_child_val
            no_improv_gens = 0
        else:
            no_improv_gens += 1

        history.append((evals_total, theta, float(best_val)))
        if verbose:
            tag = "↑" if improved else "·"
            print(
                f"[gen {gens_done}] theta={theta:.3g} fitness={best_val:.6f} {tag} "
                f"evals={evals_total}/{1+child_budget}"
            )

        if patience is not None and no_improv_gens >= patience:
            if verbose:
                print(
                    f"[stop] patience reached: {patience} generations without improvement"
                )
            break

    result: Dict[str, Any] = {
        "best_perm": parent,
        "best_fitness": float(best_val),
        "evals": evals_total,  # includes the initial parent evaluation
        "gens": gens_done,  # actually executed generations
        "history": history,
    }

    if return_routes:
        # be robust to fitness() shape; fall back to explicit *_and_ans
        val = fitness(cvrp, parent)
        if isinstance(val, tuple) and len(val) == 2:
            cost, routes = val
        else:
            cost, routes = cvrp.dynamic_programming_fitness_and_ans(parent)
        result["routes"] = routes
        result["best_fitness_exact"] = float(cost)

    return result


def nn_input_fitness(prv_fitness, cur_fitness):
    C = 10
    return np.exp(C * (np.log(cur_fitness) - np.log(prv_fitness) + 1)) / np.exp(C)


def nn_input_theta(theta):
    return (theta - 1) / 100


def one_plus_lambda_ea_with_theta_control(
    cvrp,  # instance of cvrp_cpp.CVRP
    theta_control_fun: Callable[[np.ndarray], float],
    window: int = 5,
    theta_schedule_window: Optional[np.ndarray] = None,
    lambda_: int = 100,  # offspring per generation
    max_evals: int = 100_000,  # total objective evaluations budget
    mutation: Optional[MutationFn] = None,
    init_perm: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
    patience: Optional[int] = None,  # stop after this many generations w/o improvement
    return_routes: bool = False,  # if True, also return DP routes for the best perm
    verbose: bool = False,
    generations_number: Optional[int] = None,
) -> Dict[str, Any]:
    if generations_number is None or theta_control_fun is None:
        raise ValueError("Provide both generations_number and theta_schedule.")
    if theta_schedule_window is not None and window != len(theta_schedule_window):
        raise ValueError(f"{window} first parameters must be scheduled")
    if lambda_ <= 0:
        raise ValueError("lambda_ must be positive.")
    if theta_schedule_window is None:
        theta_schedule_window = np.linspace(100.0, 50.0, window)

    rng = np.random.default_rng(seed)
    n = int(cvrp.n)

    # initialize parent
    if init_perm is None:
        parent = np.arange(n, dtype=int)
        rng.shuffle(parent)
    else:
        parent = np.array(init_perm, dtype=int)
        assert is_permutation(parent, n), "init_perm must be a permutation of 0..n-1"

    # evaluate parent (total eval counter includes this one)
    best_val = fitness(cvrp, parent)
    evals_total = 1
    evals_children = 0
    child_budget = int(min(max_evals, lambda_ * generations_number))

    history: List[Tuple[int, float, float]] = [(evals_total, -1.0, float(best_val))]
    gens_done = 0
    no_improv_gens = 0

    if verbose:
        print(f"[init] fitness={best_val:.6f}")

    progress = fq.NumpyFixedQueue(maxlen=2 * window)

    # main loop over generations with scheduled theta
    for g in range(generations_number):
        if evals_children >= child_budget:
            break
        gens_done += 1

        if g < window:
            theta = float(theta_schedule_window[g])
        else:
            theta = theta_control_fun(progress.to_numpy())

        best_child_val = None
        best_child = None

        if mutation is None:
            mut = lambda p: mutate_shift_swap_fast_fast(p, rng, theta)
        else:
            mut = lambda p: mutation(p, rng, theta)

        # generate λ offspring for this generation (respect eval budget)
        for _ in range(lambda_):
            if evals_children >= child_budget:
                break
            # mutate with current generation's theta

            child = mut(parent)

            child_val = fitness(cvrp, child)
            evals_children += 1
            evals_total += 1

            if (best_child_val is None) or (child_val < best_child_val):
                best_child_val = child_val
                best_child = child

        # update params tracking window
        progress.append(nn_input_fitness(best_val, best_child_val))
        progress.append(nn_input_theta(theta))
        # elitist selection (ties keep parent)
        improved = (best_child_val is not None) and (best_child_val < best_val)
        if improved:
            parent = best_child
            best_val = best_child_val
            no_improv_gens = 0
        else:
            no_improv_gens += 1

        history.append((evals_total, theta, float(best_val)))
        if verbose:
            tag = "↑" if improved else "·"
            print(
                f"[gen {gens_done}] theta={theta:.3g} fitness={best_val:.6f} {tag} "
                f"evals={evals_total}/{1+child_budget}"
            )

        if patience is not None and no_improv_gens >= patience:
            if verbose:
                print(
                    f"[stop] patience reached: {patience} generations without improvement"
                )
            break

    result: Dict[str, Any] = {
        "best_perm": parent,
        "best_fitness": float(best_val),
        "evals": evals_total,  # includes the initial parent evaluation
        "gens": gens_done,  # actually executed generations
        "history": history,
    }

    if return_routes:
        # be robust to fitness() shape; fall back to explicit *_and_ans
        val = fitness(cvrp, parent)
        if isinstance(val, tuple) and len(val) == 2:
            cost, routes = val
        else:
            cost, routes = cvrp.dynamic_programming_fitness_and_ans(parent)
        result["routes"] = routes
        result["best_fitness_exact"] = float(cost)

    return result


def descent(
    cvrp,  # instance of cvrp_cpp.CVRP
    lambda_: int = 1,  # для совместимости с one_plus_lambda_ea (здесь не критичен)
    max_evals: int = 100_000,  # бюджет на вызовы fitness
    mutation: Optional[MutationFn] = None,  # элементарная мутация
    init_perm: Optional[Sequence[int]] = None,  # начальная перестановка
    seed: Optional[int] = None,  # сид генератора
    patience: Optional[int] = None,  # останов после стольких итераций без улучшений
    return_routes: bool = False,  # вернуть ещё и маршруты DP
    verbose: bool = False,
    theta: float = 1.0,
    target_fitness_value: float = -np.inf
) -> Dict[str, Any]:
    """
    Локальный спуск (1+1) c многократной мутацией (через mutate_shift).
    Интерфейс и формат результата совпадают c one_plus_lambda_ea.
    """
    rng = np.random.default_rng(seed)
    n = int(cvrp.n)

    # базовая мутация: по умолчанию 2-opt
    if mutation is None:
        mut = lambda p, rng_: mutate_shift_swap_fast_fast(p, rng_, theta)
    else:
        mut = lambda p, rng_: mutation(p, rng_, theta)

    # инициализация родителя
    if init_perm is None:
        parent = np.arange(n, dtype=int)
        rng.shuffle(parent)
    else:
        parent = np.array(init_perm, dtype=int)
        assert is_permutation(parent, n), "init_perm must be a permutation of 0..n-1"

    # оценка родителя
    best_val = fitness(cvrp, parent)
    evals = 1
    gens = 0
    no_improv_gens = 0
    history: List[Tuple[int, float, float]] = [(evals, -1.0, float(best_val))]

    if verbose:
        print(f"[init] fitness={best_val:.6f}")

    # основной цикл
    while evals < max_evals:
        if best_val < target_fitness_value:
            break

        gens += 1

        # генерируем одного потомка (классический (1+1)-EA / hill-climbing)
        child = mut(parent, rng)
        child_val = fitness(cvrp, child)
        evals += 1

        improved = child_val < best_val
        if improved:
            parent = child
            best_val = child_val
            no_improv_gens = 0
        else:
            no_improv_gens += 1
            # как в исходной версии: при равенстве стоимости — принимаем перестановку (боковой шаг)
            if child_val == best_val:
                parent = child

        history.append((evals, theta, float(best_val)))

        if verbose:
            tag = "↑" if improved else ("=" if child_val == best_val else "·")
            print(f"[it {gens}] fitness={best_val:.6f} {tag} evals={evals}")

        if patience is not None and no_improv_gens >= patience:
            if verbose:
                print(
                    f"[stop] patience reached: {patience} iterations without improvement"
                )
            break

    result: Dict[str, Any] = {
        "best_perm": parent,
        "best_fitness": float(best_val),
        "evals": evals,
        "gens": gens,
        "history": history,
    }

    if return_routes:
        _, routes = cvrp.dynamic_programming_fitness_and_ans(parent)
        result["routes"] = routes

    return result


def profile():
    import cProfile
    import pstats
    from pathlib import Path

    prof_path = Path("one_plus_lambda.prof")
    with cProfile.Profile() as pr:
        dataset_dir = Path(pao.__file__).resolve().parent
        instance = pao.ProblemInstance(pao.ET.parse(dataset_dir / "X-n1001-k43.xml"))
        cvrp = cc.CVRP(
            np.asarray(instance.depot_dist),
            np.asarray(instance.dist),
            np.asarray(instance.node_q),
            np.asarray(instance.capacity),
        )

        gen = 4
        lambda_ = 2 * 10**4
        res = one_plus_lambda_ea_with_theta_schedule(
            cvrp,
            seed=42,
            lambda_=lambda_,
            max_evals=gen * lambda_,
            generations_number=gen,
            theta_schedule=np.linspace(100.0, 80.0, gen),
            verbose=True,
        )
        print(res["best_fitness"])
        cost, routes = cvrp.dynamic_programming_fitness_and_ans(res["best_perm"])
        print(len(routes))

    pr.dump_stats(str(prof_path))
    print(f"Saved profile to {prof_path}. To visualize: snakeviz {prof_path}")


if __name__ == "__main__":
    dataset_dir = Path(pao.__file__).resolve().parent
    instance = pao.ProblemInstance(pao.ET.parse(dataset_dir / "X-n101-k25.xml"))
    cvrp = cc.CVRP(
        np.asarray(instance.depot_dist),
        np.asarray(instance.dist),
        np.asarray(instance.node_q),
        np.asarray(instance.capacity),
    )
    res = one_plus_lambda_ea(cvrp, seed=42, lambda_=2 * 10**3, max_evals=10**5)
    print(res["best_fitness"])
    cost, routes = cvrp.dynamic_programming_fitness_and_ans(res["best_perm"], -1.0)
    print(len(routes))
