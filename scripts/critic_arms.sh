#!/usr/bin/env bash
# Train the four critic arms on one shared buffer/feature cache, then log gradient diagnostics.
set -euo pipefail
cd "$(dirname "$0")/.."
export TMPDIR=/tmp

BASE=spatial_v5_h10
ARMS=("min:configs/libero_spatial.yaml"
      "mean:configs/arms/critic_value_mean.yaml"
      "prop:configs/arms/critic_proprio.yaml"
      "meanprop:configs/arms/critic_value_mean_proprio.yaml")

if [ ! -f "artifacts/$BASE/features_full.pt" ]; then
  echo "missing artifacts/$BASE/features_full.pt; wait for train_rl_token" >&2
  exit 1
fi

gpu=0
pids=()
for arm in "${ARMS[@]}"; do
  name="${arm%%:*}"
  config="${arm#*:}"
  run="$BASE"
  if [ "$name" != "min" ]; then
    run="spatial_v5_$name"
    mkdir -p "artifacts/$run"
    ln -sfn "../$BASE/buffer" "artifacts/$run/buffer"
    ln -sf "../$BASE/rl_token_full.pt" "artifacts/$run/rl_token_full.pt"
    ln -sf "../$BASE/features_full.pt" "artifacts/$run/features_full.pt"
  fi
  (
    export CUDA_VISIBLE_DEVICES=$((gpu % 4))
    bash run.sh train_critic --config "$config" --run "$run" --tag full \
      > "artifacts/$run/critic_$name.log" 2>&1
    bash run.sh diagnose_critic --config "$config" --run "$run" --tag full \
      >> "artifacts/$run/critic_$name.log" 2>&1
  ) &
  pids+=($!)
  gpu=$((gpu + 1))
done
echo "critic arms: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for arm in "${ARMS[@]}"; do
  name="${arm%%:*}"
  run="$BASE"
  if [ "$name" != "min" ]; then run="spatial_v5_$name"; fi
  echo "--- $name ($run) ---"
  cat "artifacts/$run/critic_diagnostics_full.json" 2>/dev/null || echo "no diagnostics"
done
exit "$fail"
