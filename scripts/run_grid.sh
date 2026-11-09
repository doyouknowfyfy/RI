#!/usr/bin/env bash
# Run independent experiments in parallel, one per GPU (no communication between them).
# Best use of 8 GPUs for tuning: Tweedie power, seeds, fusion mode, model size ...
#
#   bash scripts/run_grid.sh data/prepared runs/grid
#   EXTRA_ARGS="--epochs 60 --scheduler cosine" bash scripts/run_grid.sh data/prepared runs/grid
#
# Each line of CONFIGS is one run; runs are assigned round-robin to the GPUs, and at most
# one run per GPU executes at a time.
set -euo pipefail

DATA=${1:-synthetic}
OUT=${2:-runs/grid}
N_GPUS=${N_GPUS:-$(nvidia-smi -L | wc -l)}
EXTRA_ARGS=${EXTRA_ARGS:-}            # appended to every run
PYTHON=${PYTHON:-python3}

CONFIGS=(
  "--loss tweedie --tweedie-power 1.2"
  "--loss tweedie --tweedie-power 1.5"
  "--loss tweedie --tweedie-power 1.8"
  "--loss tweedie --tweedie-power 2.0"
  "--loss tweedie --tweedie-power 2.5"
  "--loss mse"
  "--loss tweedie --tweedie-power 1.5 --fusion early"
  "--loss tweedie --tweedie-power 1.5 --fusion hybrid"
)

mkdir -p "$OUT"
for gpu in $(seq 0 $((N_GPUS - 1))); do
  (
    for i in $(seq "$gpu" "$N_GPUS" $((${#CONFIGS[@]} - 1))); do
      name="run$(printf %02d "$i")"
      echo "GPU $gpu -> $name: ${CONFIGS[$i]}"
      echo "${CONFIGS[$i]} $EXTRA_ARGS" > "$OUT/$name.args"
      # CUDA_VISIBLE_DEVICES makes this GPU appear as cuda:0 inside the run
      CUDA_VISIBLE_DEVICES=$gpu $PYTHON scripts/train.py --data "$DATA" \
        --output-dir "$OUT/$name" ${CONFIGS[$i]} $EXTRA_ARGS > "$OUT/$name.log" 2>&1
    done
  ) &
done
wait
echo "all runs finished (failed runs: see $OUT/runXX.log)"
$PYTHON - "$OUT" <<'PY'
import glob, json, os, sys
import pandas as pd
rows = []
for d in sorted(glob.glob(os.path.join(sys.argv[1], "run*/"))):
    name = os.path.basename(d.rstrip("/"))
    args = open(d.rstrip("/") + ".args").read().strip()
    s = os.path.join(d, "summary.json")
    rows.append({"run": name, "args": args, **(json.load(open(s)) if os.path.exists(s) else {"status": "FAILED"})})
print(pd.DataFrame(rows).sort_values("rmse" if any("rmse" in r for r in rows) else "run").to_string(index=False))
PY
