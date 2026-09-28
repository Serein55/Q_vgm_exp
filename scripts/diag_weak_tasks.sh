#!/usr/bin/env bash
# Failure-mode diagnosis of the frozen SFT baseline on LIBERO-Spatial's weak tasks.
# Two sequential eval-only runs (no training: optim.lr=0, save_interval=-1, max_steps=1):
#   t9  -> task 9 only,  50 states, 2 GPUs (0-1), 25 envs/worker
#   t56 -> tasks 5 and 6, 100 states, 4 GPUs (0-3), 25 envs/worker
# Sequential rather than concurrent because two Ray heads on one node fight over ports.
# Videos get task_id/trial_id/success_once burned into the frames (see the configs).
set -u

RLINF=/pfs/pfs-oHNwH0/ganrenda/RLinf
QVGM=/pfs/pfs-oHNwH0/ganrenda/Q_vgm
PY=$RLINF/.venv-openpi-robotwin/bin/python

run() {
  name=$1
  mkdir -p "$QVGM/artifacts/diag/$name"
  echo "[diag] $(date '+%F %T') start $name"
  (
    cd "$RLINF" || exit 1
    env TMPDIR=/tmp \
      RAY_memory_usage_threshold=0.98 \
      EMBODIED_PATH=$RLINF/examples/embodiment \
      REPO_PATH=$RLINF \
      PYTHONPATH=$RLINF \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard \
      $PY examples/embodiment/train_embodied_agent.py \
      --config-path $QVGM/configs/diag \
      --config-name "libero_spatial_sft_diag_$name" \
      runner.logger.log_path="$QVGM/artifacts/diag/$name" \
      >"$QVGM/artifacts/diag/$name.log" 2>&1
  )
  rc=$?
  echo "[diag] $(date '+%F %T') $name exit rc=$rc"
}

run t9
run t56
echo "[diag] $(date '+%F %T') all done"
