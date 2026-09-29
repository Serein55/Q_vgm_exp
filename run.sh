#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SCRIPT="${1:?Usage: bash Q_vgm/run.sh eval_sft [options]}"
shift
case "$SCRIPT" in
  collect_rollouts|eval_qvgm|eval_sft|preflight|train_critic|train_offline_qvgm|train_rl_token) ;;
  *) echo "Unknown script: $SCRIPT" >&2; exit 2 ;;
esac
CONFIG="configs/libero_spatial_checkpoint.yaml"
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[i]}" == "--config" ]]; then CONFIG="${args[i+1]}"; fi
done
PYTHON_BIN=$(python - "$CONFIG" <<'PY'
import sys
from qvgm.config import load_config
print(load_config(sys.argv[1])['paths']['python'])
PY
)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
exec "$PYTHON_BIN" "scripts/$SCRIPT.py" "$@"
