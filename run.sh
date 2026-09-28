#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SCRIPT="${1:?Usage: bash Q_vgm/run.sh eval_sft [options]}"
shift
case "$SCRIPT" in
  recovery_candidates|recovery_fit|recovery_watch|eval_sft|eval_qvgm|preflight|collect_rollouts|train_rl_token|train_critic|train_offline_qvgm|offline_pipeline|diagnose_actor|diagnose_critic|diagnose_critic_support|check_guidance_environment|check_paper_contract|diagnose_task1|diag_grasp_trace|counterfactual_ab|counterfactual_report|guidance_direction|controller_coverage_pipeline|train_controller_critic|controller_training_report|train_candidate_policy|confirm_retrieval|retrieval_closed_loop|finish_retrieval|privileged_state|privileged_state_critic) ;;
  *) echo "Unknown script: $SCRIPT" >&2; exit 2 ;;
esac
CONFIG="configs/controller_coverage.yaml"
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
