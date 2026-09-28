#!/usr/bin/env bash
# Wait for the feature cache, train the four critic arms, smoke-test both critic state
# layouts, then run the full paired counterfactual A/B.
set -euo pipefail
cd "$(dirname "$0")/.."
export TMPDIR=/tmp

BASE=spatial_v5_h10
FEATURES="artifacts/$BASE/features_full.pt"

while [ ! -f "$FEATURES" ]; do
  if ! tmux has-session -t qvgm_rlt 2>/dev/null; then
    echo "rl_token session gone but $FEATURES missing" >&2
    tail -5 "artifacts/$BASE/rl_token_full.log" >&2
    exit 1
  fi
  sleep 30
done
echo "[stage23] $(date '+%F %T') features ready"

bash scripts/critic_arms.sh
echo "[stage23] $(date '+%F %T') critic arms done"

smoke() {  # arm config run
  local name="cf_smoke_${1}_$(date +%s)"
  CUDA_VISIBLE_DEVICES=5 bash run.sh counterfactual_ab \
    --config "$2" --run "$3" --tag full --name "$name" --task-ids 0 \
    --episodes-per-task 1 --states-per-episode 1 --repeats 1 \
    > "artifacts/$3/$name.log" 2>&1
  echo "[stage23] smoke $1:"
  cat "artifacts/$3/$name/summary.json"
}
smoke min configs/libero_spatial.yaml "$BASE"
smoke prop configs/arms/critic_proprio.yaml spatial_v5_prop
echo "[stage23] $(date '+%F %T') smokes passed"

bash scripts/counterfactual_arms.sh
echo "[stage23] $(date '+%F %T') counterfactual A/B complete"
