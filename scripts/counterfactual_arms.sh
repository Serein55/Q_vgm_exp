#!/usr/bin/env bash
# Paired counterfactual A/B for each critic arm, sharded by task across GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
export TMPDIR=/tmp

ARMS=${ARMS:-"min mean prop meanprop"}
EPISODES=${EPISODES:-2}
STATES=${STATES:-3}
REPEATS=${REPEATS:-8}
JOBS=${JOBS:-8}

for arm in $ARMS; do
  case "$arm" in
    min) config=configs/libero_spatial.yaml; run=spatial_v5_h10 ;;
    mean) config=configs/arms/critic_value_mean.yaml; run=spatial_v5_mean ;;
    prop) config=configs/arms/critic_proprio.yaml; run=spatial_v5_prop ;;
    meanprop) config=configs/arms/critic_value_mean_proprio.yaml; run=spatial_v5_meanprop ;;
    *) echo "unknown arm: $arm" >&2; exit 2 ;;
  esac
  if [ ! -f "artifacts/$run/critic_full.pt" ]; then
    echo "missing artifacts/$run/critic_full.pt; run scripts/critic_arms.sh first" >&2
    exit 1
  fi
  echo "=== arm $arm ($run) ==="
  seq 0 9 | xargs -P "$JOBS" -I@ bash -c '
    task=$1; arm=$2; config=$3; run=$4; ep=$5; st=$6; rp=$7
    name="cf_${arm}_t${task}"
    if [ -f "artifacts/${run}/${name}/status.txt" ]; then
      echo "skip ${name} (exists)"; exit 0
    fi
    CUDA_VISIBLE_DEVICES=$((task % 8)) bash run.sh counterfactual_ab \
      --config "$config" --run "$run" --tag full --name "$name" --task-ids "$task" \
      --episodes-per-task "$ep" --states-per-episode "$st" --repeats "$rp" \
      > "artifacts/${run}/${name}.log" 2>&1
  ' _ @ "$arm" "$config" "$run" "$EPISODES" "$STATES" "$REPEATS"
done

../RLinf/.venv-openpi-robotwin/bin/python scripts/counterfactual_report.py --arms $ARMS
