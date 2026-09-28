#!/usr/bin/env bash
# Stage 4 for one critic arm: 500 offline actor steps, then the 500-episode eval protocol.
#
# The eval split (tasks 0-4 / 5-9, 50 episodes each) is identical to
# scripts/eval_sft_baseline.sh, so the result is directly comparable to the 376/500 SFT
# baseline measured in this harness.
set -euo pipefail
cd "$(dirname "$0")/.."
export TMPDIR=/tmp

ARM=${1:?usage: actor_eval.sh <min|mean|prop|meanprop> [train_gpu] [eval_gpu_a] [eval_gpu_b]}
TRAIN_GPU=${2:-0}
EVAL_A=${3:-1}
EVAL_B=${4:-2}
EXTRA=("${@:5}")

case "$ARM" in
  min) config=configs/libero_spatial.yaml; run=spatial_v5_h10 ;;
  mean) config=configs/arms/critic_value_mean.yaml; run=spatial_v5_mean ;;
  prop) config=configs/arms/critic_proprio.yaml; run=spatial_v5_prop ;;
  meanprop) config=configs/arms/critic_value_mean_proprio.yaml; run=spatial_v5_meanprop ;;
  bias) config=configs/libero_spatial.yaml; run=spatial_v5_bias; EXTRA=(--bias-critic "${EXTRA[@]}") ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

if [ "$ARM" = bias ]; then
  required="artifacts/$run/guidance_direction_full.json"
else
  required="artifacts/$run/critic_full.pt"
fi
if [ ! -f "$required" ]; then
  echo "missing $required" >&2
  exit 1
fi

echo "[actor] $(date '+%F %T') arm=$ARM train gpu=$TRAIN_GPU run=$run ${EXTRA[*]:-}"
CUDA_VISIBLE_DEVICES=$TRAIN_GPU bash run.sh train_offline_qvgm \
  --config "$config" --run "$run" --tag full --steps 500 "${EXTRA[@]}" \
  > "artifacts/$run/actor_full.log" 2>&1
echo "[actor] $(date '+%F %T') training done rc=$?"

for half in a b; do
  if [ "$half" = a ]; then gpu=$EVAL_A; tasks="0 1 2 3 4"; else gpu=$EVAL_B; tasks="5 6 7 8 9"; fi
  name="qvgm_${ARM}_t${half}"
  echo "[actor] $(date '+%F %T') eval gpu=$gpu name=$name tasks=$tasks"
  CUDA_VISIBLE_DEVICES=$gpu bash run.sh eval_qvgm \
    --config "$config" --task-ids $tasks --episodes-per-task 50 \
    --actor-checkpoint "artifacts/$run/actor_full.pt" --name "$name" \
    > "artifacts/eval/$name.log" 2>&1 &
done
wait
echo "[actor] $(date '+%F %T') arm=$ARM complete"
../RLinf/.venv-openpi-robotwin/bin/python - "$ARM" <<'PY'
import collections
import json
import sys
from pathlib import Path

arm = sys.argv[1]
total = successes = 0
per = collections.Counter()
counted = collections.Counter()
for path in sorted(Path("artifacts/eval").glob(f"qvgm_{arm}_t*/episodes.jsonl")):
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            successes += bool(row["success"])
            per[row["task_id"]] += bool(row["success"])
            counted[row["task_id"]] += 1
print(f"arm {arm}: {successes}/{total}" + (f" = {successes / total:.1%}" if total else ""))
print({t: f"{per[t]}/{counted[t]}" for t in sorted(counted)})
PY
