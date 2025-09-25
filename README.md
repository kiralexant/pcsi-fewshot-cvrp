<div align="center">

# PCSI-FewShot-CVRP:<br>Parameter Control Strategy Inference for<br>Few-Shot Randomized Heuristic Search<br>Applied to CVRP

</div>


<div align="center">

  <a href="https://mit-license.org/"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python version"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
  <a href="https://git.liacs.nl/antonovk/basinsattribution"><img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg" alt="Contributions welcome"></a>
</div>


## Overview (English, см. русскую версию ниже)


**PCSI-FewShot-CVRP** learns a **parameter‑control strategy** for a **few‑shot** **(1+λ) evolutionary algorithm** solving **CVRP**.  
The control policy is a **neural network** trained with **Kernel‑PCA‑assisted Bayesian Optimization (KPCA‑BO)** on problem instances, and then applied inside the few-shot (1+λ) evolutionary algorithm to new instances.  

<div align="center">

![I_target](imgs/train-test-instances-post.png)

</div>

Figure: *Performance of the $(1+\lambda)$-EA with the proposed $\theta$-control (blue) vs. state-independent/static baselines: linear schedule $\theta: 100\!\to\!1$ (orange), fixed $\theta=1$ (green), fixed $\theta=10$ (red). Curves show best-so-far objective $f(\bm{x})$ vs. number of function evaluations ($\times 10^6$). Shaded ribbons denote **standard deviation** across runs. Each algorithm was run **100 times**. Green checkmarks indicate cases where the proposed method achieves the best performance within the given budget. Top row: training instances. Bottom row: previously unseen test instances.*


> ℹ️ The (1+λ)-EA is **few-shot**, meaning that it has large λ and works in few generations to exploit parallel hardware.

> ℹ️ The (1+λ)-EA here serves as an **illustrative proxy** of a practical EA. The repository emphasizes **policy inference** for CVRP; the EA is intentionally simple to keep the learning pipeline transparent.


## Key modules
- **Training** policy on instances: `FewShotCVRP/examples/params_search/nn_parallel_kpcabo.py`
- **Loading** trained policies: `FewShotCVRP/dataset/theta_control_loader.py`
- **Few‑shot solver** using a learned policy: `FewShotCVRP/few_shot.py`

**Dependencies** (declared in `pyproject.toml`): `numpy`, `numba`, `pytest`, `rich`, `matplotlib`, `pandas`, `pyarrow`, `seaborn`, `scikit‑learn`, `scipy`, `torch`, `botorch`, `gpytorch`.



---

## Installation

Clone this repository and install in editable mode:

```bash
git clone --recurse-submodules https://github.com/kiralexant/pcsi-fewshot-cvrp.git
cd pcsi-fewshot-cvrp
python -m pip install -e .
```

Rebuild only the C++ extension (if you changed just the C++ code):

```bash
python setup.py build_ext --inplace
```

Initialize submodules:

```bash
git submodule update --init --recursive
```

> Python **≥3.10** is required. GPU is supported where available by the libraries you install (see `pyproject.toml`).

---

## Project layout (selected files)

```
pcsi-fewshot-cvrp/
├─ FewShotCVRP/
│  ├─ examples/
│  │  └─ params_search/
│  │     └─ nn_parallel_kpcabo.py    # Train θ‑control policy via KPCA‑BO (no CLI args; configure in-file)
│  ├─ dataset/
│  │  └─ theta_control_loader.py     # Interface to load/use trained policies
│  ├─ few_shot.py                    # Few‑shot (1+λ)‑EA that uses a learned policy
│  └─ ...
├─ tests/
├─ pyproject.toml
├─ setup.py
└─ README.md
```

---

## How to use

### 1) Train the parameter‑control policy on the **current** CVRP setup
1. Open `FewShotCVRP/examples/params_search/nn_parallel_kpcabo.py`.
2. Configure training instances and algorithm settings **inside the file** (there are no command‑line arguments).
3. Run it as a module (no options needed):
   ```bash
   cd FewShotCVRP/examples
   bash runbg.sh -- python params_search/nn_parallel_kpcabo.py
   ```
   This will start the KPCA‑BO training loop using the in‑file configuration. Check the run directory/logs as configured inside the script.

### 2) Apply a trained policy in the **few‑shot** (1+λ)‑EA solver
1. Ensure your trained policy is saved (see how checkpoints are written in the training script).
2. Use the loader API from `FewShotCVRP/dataset/theta_control_loader.py` to obtain the policy object.
3. Use `FewShotCVRP/few_shot.py` to run the few‑shot solver that consumes the loaded policy.  
   You can run it as a module (if it defines a default demo) or import it and call the function/class provided in the file:
   ```python
   res = few_shot.few_shot_optimization(
      "X-n641-k35.xml",
      random_seed=2025,
      numproc=50,
      subsequent_local_opt=True,
      local_opt_budget_fraction=0.01,
   )
   ```

### 3) Adapting to a **new** CVRP dataset
- Duplicate/modify the configuration block inside `nn_parallel_kpcabo.py` to point to your new dataset/instances.
- Re‑run the training module as above.  
- Use the resulting policy with `few_shot.py` following the same pattern.

> **Few‑shot regime:** use a large λ and a small number of generations; the learned policy handles rapid adaptation.

---

## Reproducibility & performance
- Random seeds are fixed inside scripts to make runs reproducible.
- The codebase includes C++ kernels / JIT'ed hot paths (see source files) to accelerate heavy parts.
- GPU support depends on your installed PyTorch/BoTorch/GPyTorch builds.

---

## Citing

If you use this repository, please cite:

- Antonov, K., Raponi, E., Wang, H., & Doerr, C. **High Dimensional Bayesian Optimization with Kernel Principal Component Analysis**. _PPSN XVII_, Springer, 2022, 118–131.  

---

## License

**MIT License**

---

# 🇷🇺 Русская версия

**PCSI-FewShot-CVRP** обучает **стратегию управления параметрами** для **few-shot** **(1+λ) эволюционного алгоритма**, решающего **CVRP**.  
Стратегия управления — это **нейронная сеть**, обучаемая с помощью **байесовской оптимизации с Kernel-PCA (KPCA-BO)** на инстансах задачи и затем применяемая внутри few-shot (1+λ) эволюционного алгоритма к новым инстансам.  

<div align="center">

![I_target](imgs/train-test-instances-post.png)

</div>

Рисунок: *Производительность $(1+\lambda)$-ЭА с предлагаемым управлением $\theta$ (синий) по сравнению с базовыми, не зависящими от состояния, стратегиями: линейное расписание $\theta: 100\!\to\!1$ (оранжевый), фиксированное $\theta=1$ (зелёный), фиксированное $\theta=10$ (красный). Показано лучшее на текущий момент $f(\bm{x})$ в зависимости от числа вычислений ($\times 10^6$). Полупрозрачные ленты — **стандартное отклонение** по запускам. Для каждого алгоритма выполнено **100 запусков**. Зелёные галочки отмечают случаи, когда предлагаемый метод даёт лучший результат при заданном бюджете. Верхний ряд — обучающие инстансы, нижний — ранее неизвестные модели тестовые инстансы.*

> ℹ️ (1+λ)-EA — **few-shot**, то есть использует большое λ и работает в малом числе поколений, чтобы эффективно задействовать параллельные вычислительные ресурсы.

> ℹ️ (1+λ)-EA здесь служит **иллюстративным примером** практического ЭА. В репозитории основной акцент сделан на **выводе стратегии (policy inference)** для CVRP; сам ЭА намеренно упрощён, чтобы сделать процесс обучения прозрачным.

---

## Ключевые модули
- **Обучение** стратегии: `FewShotCVRP/examples/params_search/nn_parallel_kpcabo.py`
- **Загрузка** обученных стратегий: `FewShotCVRP/dataset/theta_control_loader.py`
- **Few‑shot решатель**: `FewShotCVRP/few_shot.py`

**Зависимости** (в `pyproject.toml`): `numpy`, `numba`, `pytest`, `rich`, `matplotlib`, `pandas`, `pyarrow`, `seaborn`, `scikit‑learn`, `scipy`, `torch`, `botorch`, `gpytorch`.

---

## Установка

```bash
git clone --recurse-submodules https://github.com/kiralexant/pcsi-fewshot-cvrp.git
cd pcsi-fewshot-cvrp
python -m pip install -e .
```

Пересборка только C++‑расширения:

```bash
python setup.py build_ext --inplace
```

Инициализация submodules:

```bash
git submodule update --init --recursive
```

> Нужен Python **≥3.10**. Поддержка GPU — в зависимости от установленных библиотек (см. `pyproject.toml`).

---

## Структура проекта (выборка)

```
pcsi-fewshot-cvrp/
├─ FewShotCVRP/
│  ├─ examples/
│  │  └─ params_search/
│  │     └─ nn_parallel_kpcabo.py
│  ├─ dataset/
│  │  └─ theta_control_loader.py
│  ├─ few_shot.py
│  └─ ...
├─ tests/
├─ pyproject.toml
├─ setup.py
└─ README.md
```

---

## Как пользоваться

### 1) Обучение стратегии на **текущей** задаче
1. Откройте `FewShotCVRP/examples/params_search/nn_parallel_kpcabo.py`.
2. Настройте инстансы и параметры **внутри файла**.
3. Запустите модуль:
   ```bash
   cd FewShotCVRP/examples
   bash runbg.sh -- python params_search/nn_parallel_kpcabo.py
   ```

### 2) Применение обученной стратегии в **few‑shot** решателе (1+λ)‑ЭА
1. Загрузите стратегию через `FewShotCVRP/dataset/theta_control_loader.py`.
2. Используйте `FewShotCVRP/few_shot.py` для запуска few‑shot решателя на новой задаче CVRP из датасета.
   ```python
   res = few_shot.few_shot_optimization(
      "X-n641-k35.xml",
      random_seed=2025,
      numproc=50,
      subsequent_local_opt=True,
      local_opt_budget_fraction=0.01,
   )
   ```

### 3) Новая dataset задач CVRP
- Измените конфигурацию в `nn_parallel_kpcabo.py` под новый датасет.
- Перезапустите обучение и используйте полученную стратегию с `few_shot.py`.

> **Few‑shot:** большое λ, мало поколений; быстрая адаптация за счёт выученной политики.

---

## Воспроизводимость и производительность
- Random seed фиксируются в скриптах для воспроизводимости.
- Быстрые участки реализованы на C++/JIT
- Подразумевается поддержка обучения на GPU, если он доступен.

---

## Цитирование

- Antonov, K., Raponi, E., Wang, H., & Doerr, C. **High Dimensional Bayesian Optimization with Kernel Principal Component Analysis**. _PPSN XVII_, Springer, 2022, 118–131.  

---

## Лицензия

**MIT**
