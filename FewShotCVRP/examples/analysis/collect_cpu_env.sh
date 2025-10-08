#!/usr/bin/env bash
set -euo pipefail

TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="system_report_${TS}"
mkdir -p "$OUTDIR"

# Чтобы парсить метки на английском, форсим C-локаль для системных утилит
export LC_ALL=C

# ---- Утилиты: проверка наличия ---------------------------------------------
have() { command -v "$1" >/dev/null 2>&1; }

# ---- Базовая системная инфа ------------------------------------------------
HOST="$(hostname || true)"
WHO="$(whoami || true)"
KERNEL="$(uname -r 2>/dev/null || true)"
UNAME_ALL="$(uname -a 2>/dev/null || true)"   # ядро, архитектура и т.п.

# ОС (os-release — стандартный источник идентификации дистрибутива)
OS_PRETTY="unknown"
if [ -r /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  OS_PRETTY="${PRETTY_NAME:-$NAME $VERSION}"
fi

# ---- CPU / топология --------------------------------------------------------
# lscpu даёт модель, сокеты, ядра, потоки, кеши и NUMA
if have lscpu; then
  lscpu > "$OUTDIR/lscpu.txt"
  CPU_MODEL="$(grep -E '^Model name:' "$OUTDIR/lscpu.txt" | sed 's/.*: *//')" || CPU_MODEL=""
  CPU_SOCKETS="$(grep -E '^Socket\(s\):' "$OUTDIR/lscpu.txt" | awk -F': *' '{print $2}')" || CPU_SOCKETS=""
  CPU_CORES_PER_SOCKET="$(grep -E '^Core\(s\) per socket:' "$OUTDIR/lscpu.txt" | awk -F': *' '{print $2}')" || CPU_CORES_PER_SOCKET=""
  CPU_THREADS_PER_CORE="$(grep -E '^Thread\(s\) per core:' "$OUTDIR/lscpu.txt" | awk -F': *' '{print $2}')" || CPU_THREADS_PER_CORE=""
  CPU_TOTAL="$(grep -E '^CPU\(s\):' "$OUTDIR/lscpu.txt" | head -n1 | awk -F': *' '{print $2}')" || CPU_TOTAL=""
  CPU_L3="$(grep -E '^L3 cache:' "$OUTDIR/lscpu.txt" | awk -F': *' '{print $2}')" || CPU_L3=""
  NUMA_NODES="$(grep -E '^NUMA node\(s\):' "$OUTDIR/lscpu.txt" | awk -F': *' '{print $2}')" || NUMA_NODES=""
else
  CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: *//')"
  CPU_TOTAL="$(grep -c '^processor' /proc/cpuinfo || true)"
  CPU_SOCKETS=""; CPU_CORES_PER_SOCKET=""; CPU_THREADS_PER_CORE=""; CPU_L3=""; NUMA_NODES=""
fi

# Число доступных процессорных юнитов (учитывает cgroups/SLURM ограничения)
NPROC="$( (have nproc && nproc) || true )"
NPROC_ALL="$( (have nproc && nproc --all) || true )"

# ---- Память / NUMA ----------------------------------------------------------
if have free; then
  free -h > "$OUTDIR/free.txt"
fi
if have numactl; then
  numactl --hardware > "$OUTDIR/numactl.txt" 2>/dev/null || true
fi

# ---- Планировщик и модули окружения -----------------------------------------
# SLURM: выведем ключевые переменные, если они заданы
{
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
  echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-}"
  echo "SLURM_NTASKS=${SLURM_NTASKS:-}"
  echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
  echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-}"
  echo "SLURM_MEM_PER_CPU=${SLURM_MEM_PER_CPU:-}"
  echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-}"
} > "$OUTDIR/slurm_env.txt"

# Environment Modules / Lmod (если есть)
{
  if command -v module >/dev/null 2>&1; then
    module -t list 2>&1 || true
  elif command -v ml >/dev/null 2>&1; then
    ml 2>&1 || true
  fi
} > "$OUTDIR/modules_list.txt"

# ---- Компиляторы -------------------------------------------------------------
{
  (gcc --version 2>/dev/null | head -n1) || true
  (g++ --version 2>/dev/null | head -n1) || true
  (gfortran --version 2>/dev/null | head -n1) || true
  (clang --version 2>/dev/null | head -n1) || true
} > "$OUTDIR/compilers.txt"

# ---- Python / NumPy / BLAS / OpenMP -----------------------------------------
PY_REPORT="$OUTDIR/python_env.txt"
python3 - <<'PY' > "$PY_REPORT" 2>&1 || true
import sys, os, json, platform
print("Python:", sys.version.replace("\n"," "))
print("Executable:", sys.executable)
# Пакеты (если стоят)
def ver(m):
    try:
        mod = __import__(m)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not installed"
mods = ["numpy", "scipy", "pandas", "sklearn", "numexpr"]
for m in mods:
    print(f"{m}: {ver(m)}")
# BLAS/LAPACK backend из NumPy
try:
    import numpy as np
    try:
        # NumPy >= 2.0
        cfg = np.show_config(mode="dicts")  # type: ignore
        print("NumPy show_config (dict):", json.dumps(cfg, indent=2, default=str))
    except TypeError:
        # Старые версии: перехват stdout
        import io, contextlib
        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            np.__config__.show()
        print("NumPy show_config:\n" + s.getvalue())
except Exception as e:
    print("NumPy config error:", e)

# Популярные тред-переменные
for k in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"]:
    print(f"{k}={os.environ.get(k,'')}")
# glibc
try:
    import subprocess
    out = subprocess.check_output(["ldd","--version"], text=True, stderr=subprocess.STDOUT).splitlines()[0]
    print("ldd:", out)
except Exception:
    pass
PY

# ---- Короткий LaTeX-фрагмент для статьи -------------------------------------
LAT="$OUTDIR/hardware_snippet.tex"
{
  echo "% Autogenerated on $(date)"
  echo "\\begin{itemize}"
  echo "  \\item \\textbf{Узел:} ${HOST} (пользователь: ${WHO})"
  echo "  \\item \\textbf{ОС:} ${OS_PRETTY}, ядро ${KERNEL}"
  echo "  \\item \\textbf{CPU:} ${CPU_MODEL}"
  if [ -n "${CPU_SOCKETS}${CPU_CORES_PER_SOCKET}${CPU_THREADS_PER_CORE}" ]; then
    echo "  \\item \\textbf{Топология:} сокетов=${CPU_SOCKETS}, ядер/сокет=${CPU_CORES_PER_SOCKET}, потоков/ядро=${CPU_THREADS_PER_CORE} (всего логических CPU=${CPU_TOTAL})"
  else
    echo "  \\item \\textbf{Логических CPU:} ${CPU_TOTAL}"
  fi
  if [ -n "${CPU_L3}" ]; then
    echo "  \\item \\textbf{L3 Cache:} ${CPU_L3}"
  fi
  if [ -n "${NUMA_NODES}" ]; then
    echo "  \\item \\textbf{NUMA:} узлов=${NUMA_NODES}"
  fi
  echo "  \\item \\textbf{Доступные процессорные юниты (cgroups/SLURM):} $(echo ${NPROC:-}) из $(echo ${NPROC_ALL:-})"
  echo "  \\item \\textbf{Примечание:} эксперименты выполнялись \\emph{только на CPU} (без использования GPU)."
  echo "\\end{itemize}"
} > "$LAT"

# ---- Итоговый краткий отчёт в STDOUT ----------------------------------------
cat <<EOF
==================== CPU-only Experiment Environment ====================
Host         : ${HOST}
User         : ${WHO}
OS           : ${OS_PRETTY}
Kernel       : ${KERNEL}
Architecture : $(uname -m 2>/dev/null || true)

CPU Model    : ${CPU_MODEL}
Sockets      : ${CPU_SOCKETS}
Cores/socket : ${CPU_CORES_PER_SOCKET}
Threads/core : ${CPU_THREADS_PER_CORE}
Logical CPUs : ${CPU_TOTAL}
L3 Cache     : ${CPU_L3}
NUMA nodes   : ${NUMA_NODES}
nproc (avail): ${NPROC}    | nproc --all: ${NPROC_ALL}

Key files saved to: ${OUTDIR}/
  - lscpu.txt, free.txt, numactl.txt, slurm_env.txt, modules_list.txt
  - compilers.txt, python_env.txt
  - hardware_snippet.tex  (готовая LaTeX-вставка)

Note: CPU-only pipeline; GPUs were not used.
==========================================================================

EOF
