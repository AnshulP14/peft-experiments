#!/usr/bin/env bash
# Overnight large-scale experimentation: launch (or resume) the Optuna study
# with a wall-clock budget, detached, logging to output/tuning/<study>/.log
#
#   ./run_overnight.sh                     # 150 trials or 12h, default env
#   PEFT_MODEL=Qwen/Qwen2.5-1.5B-Instruct N_TRIALS=100 TIMEOUT=10h ./run_overnight.sh
#   ./run_overnight.sh                    # re-run in the morning: auto-resumes
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-../.venv/bin/python}"
STUDY="${STUDY:-$(date +%m%d-%H%M%S)-overnight}"
N_TRIALS="${N_TRIALS:-150}"
TIMEOUT="${TIMEOUT:-12h}"
REPEATS="${REPEATS:-3}"
LOG_DIR="output/tuning/$STUDY"
mkdir -p "$LOG_DIR"

# sanity: mlflow server reachable?
if ! curl -sf -o /dev/null http://127.0.0.1:5000/; then
  echo "starting mlflow server..."
  nohup "$PY" -m mlflow server --backend-store-uri "sqlite://$PWD/output/mlruns.db" \
    --default-artifact-root "$PWD/output/mlflow-artifacts" \
    --host 127.0.0.1 --port 5000 > output/mlflow-server.log 2>&1 &
  sleep 8
fi

echo "launching study $STUDY: $N_TRIALS trials, budget $TIMEOUT"
nohup "$PY" tune.py \
  --study "$STUDY" \
  --n_trials "$N_TRIALS" \
  --timeout "$TIMEOUT" \
  --repeats "$REPEATS" \
  > "$LOG_DIR/run.log" 2>&1 &

echo "pid: $!  log: $LOG_DIR/run.log"
echo "monitor: tail -f $LOG_DIR/run.log   |   UI: http://127.0.0.1:5000"
echo "kill & resume: kill $!  then  STUDY=$STUDY ./run_overnight.sh"
