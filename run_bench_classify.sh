#!/usr/bin/env bash
# Large-scale Kev vs Laya label-prediction benchmark (MHDialog + IndieMH) at 4/8/16-bit, detached. Two phases:
#   1. accuracy: every (model, bits) config over all examples, JOBS configs at a time (each its own process);
#      output/bench_classify/<run>/preds/<model>-<bits>bit.jsonl, resumed on re-run
#   2. speed: one config at a time, alone on the machine, SPEED_N examples per dataset -> <run>/speed/
# summary.md takes quality from phase 1 and latency from phase 2 (from phase 1 until phase 2 has run).
#
#   ./run_bench_classify.sh                          # new run, full grid
#   RUN=1019-1644-bench ./run_bench_classify.sh      # resume that run
#   MODELS="laya kev-0.8b" BITS="16" JOBS=1 ./run_bench_classify.sh
#   MODELS="kev-4b" SPLITS="test" JOBS=1 ./run_bench_classify.sh   # Kev-4B: ~5 s/example on an M2 16 GB (swaps)
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-.venv-bench/bin/python}"
RUN="${RUN:-$(date +%m%d-%H%M)-bench}"
MODELS="${MODELS:-laya kev-0.8b}"
BITS="${BITS:-16 8 4}"
SPLITS="${SPLITS:-train val test}"
JOBS="${JOBS:-1}"   # concurrent configs in phase 1. 1: on an M2 one config already saturates the GPU; 3 Laya configs together ran at 0.88 ex/s vs 1.67 alone, and swapped
SPEED_N="${SPEED_N:-300}"
SPEED_SPLITS="${SPEED_SPLITS:-test val}"
OUT="output/bench_classify/$RUN"
mkdir -p "$OUT/logs"

configs=""
for m in $MODELS; do for b in $BITS; do configs+="$m $b"$'\n'; done; done
export PY OUT SPLITS JOBS SPEED_N SPEED_SPLITS CONFIGS="$configs"

nohup bash -c '
  echo "=== phase 1 (accuracy), $JOBS at a time  $(date)"
  printf "%s" "$CONFIGS" | xargs -P "$JOBS" -L 1 bash -c '"'"'
    echo "--- start $0 ${1}bit  $(date)"
    "$PY" scripts/bench_classify.py --model "$0" --bits "$1" --splits $SPLITS --out "$OUT" >> "$OUT/logs/$0-${1}bit.log" 2>&1 \
      && echo "--- done $0 ${1}bit  $(date)" || echo "!!! $0 ${1}bit failed, see $OUT/logs/$0-${1}bit.log"
    "$PY" scripts/bench_classify.py --summarize --out "$OUT" > /dev/null
  '"'"'
  echo "=== phase 2 (speed, solo, $SPEED_N per dataset)  $(date)"
  printf "%s" "$CONFIGS" | while read -r m b; do
    rm -f "$OUT/speed/$m-${b}bit.jsonl"
    "$PY" scripts/bench_classify.py --model "$m" --bits "$b" --splits $SPEED_SPLITS --limit "$SPEED_N" --subdir speed --out "$OUT" \
      >> "$OUT/logs/speed-$m-${b}bit.log" 2>&1 && echo "--- speed $m ${b}bit  $(date)" || echo "!!! speed $m ${b}bit failed"
  done
  "$PY" scripts/bench_classify.py --summarize --out "$OUT" > /dev/null
  echo "=== all done $(date)"
' >> "$OUT/run.log" 2>&1 &
PID=$!

echo "run: $RUN   pid: $PID   configs: $(printf "%s" "$configs" | tr '\n' ',')"
echo "log:     tail -f $OUT/run.log   (per config: $OUT/logs/)"
echo "results: $OUT/summary.md (refreshed after every config)"
echo "resume:  pkill -f 'bench_classify.py --model'; kill $PID  then  RUN=$RUN ./run_bench_classify.sh"
