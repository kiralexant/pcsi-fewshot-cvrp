#!/usr/bin/env bash
# Run a command with nohup in background, keep stdout/stderr logs and a PID file.
# Usage:
#   ./runbg.sh [--logdir DIR] [--cd DIR] [--env FILE] -- <command and args...>
#
# Examples:
#   ./runbg.sh --logdir "$HOME/bglogs" -- python train.py --epochs 50
#   ./runbg.sh --cd /data/job1 --env ~/.myenvs/activate.sh -- ./run_many.sh --fast

set -Eeuo pipefail

LOGDIR="./outputs/bglogs"
CD=""
ENV_FILE=""

# -------- Parse options --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --logdir) LOGDIR="$2"; shift 2 ;;
    --cd)     CD="$2";     shift 2 ;;
    --env)    ENV_FILE="$2"; shift 2 ;;
    --) shift; break ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--logdir DIR] [--cd DIR] [--env FILE] -- <command...>" >&2
      exit 1
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "No command provided. Usage: $0 ... -- <command...>" >&2
  exit 1
fi

mkdir -p "$LOGDIR"

# Make names for logs/PID
ts="$(date +%Y%m%d_%H%M%S)"
cmd_slug="$(printf '%s' "$*" | tr -cs '[:alnum:]._+-' '_' | cut -c1-60)"
LOG_OUT="$LOGDIR/${ts}_${cmd_slug}.out.log"
LOG_ERR="$LOGDIR/${ts}_${cmd_slug}.err.log"
PIDFILE="$LOGDIR/${ts}_${cmd_slug}.pid"

# Build inline script for bash -lc (no temp file is created)
RUN='set -Eeuo pipefail'
if [[ -n "$CD" ]]; then
  RUN+=$'\n''cd '"'"$CD"'" || exit 1'
fi
if [[ -n "$ENV_FILE" ]]; then
  RUN+=$'\n''source '"'"$ENV_FILE"'"'
fi
RUN+=$'\n'"$*"

# Start with nohup; redirect stdout/stderr to logs; run in background
nohup bash -c "$RUN" </dev/null >"$LOG_OUT" 2>"$LOG_ERR" &

PID=$!
printf '%s\n' "$PID" >"$PIDFILE"

echo "Started PID:   $PID"
echo "stdout log:    $LOG_OUT"
echo "stderr log:    $LOG_ERR"
echo "pid file:      $PIDFILE"
echo
echo "Tips:"
echo "  tail -f '$LOG_OUT'    # смотреть stdout"
echo "  tail -f '$LOG_ERR'    # смотреть stderr"
echo "  kill \$(cat '$PIDFILE')   # остановить процесс"
