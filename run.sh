#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SCRIPT="${1:?Usage: bash Q_vgm/run.sh eval_sft [options]}"
shift
case "$SCRIPT" in
  recovery_candidates|recovery_fit|eval_sft|preflight|collect_rollouts|train_rl_token|train_critic|train_offline_qvgm|eval_qvgm|offline_pipeline|diagnose_actor|diagnose_critic|diagnose_critic_support|check_guidance_environment|check_paper_contract|diagnose_task1) ;;
  *) echo "Unknown script: $SCRIPT" >&2; exit 2 ;;
esac
CONFIG="configs/libero_spatial.yaml"
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
