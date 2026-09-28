#!/usr/bin/env bash
# Parallel re-collection of the 150-episode offline buffer (h10 released conditioning).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=configs/libero_spatial.yaml
RUN=spatial_v5_h10
ROOT_DIR="artifacts/$RUN"
mkdir -p "$ROOT_DIR/buffer"

export TMPDIR=/tmp

# Single writer for the collection manifest so concurrent collectors cannot race on it.
../RLinf/.venv-openpi-robotwin/bin/python - "$CONFIG" "$ROOT_DIR/buffer/config.json" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from qvgm.config import load_config

dest = Path(sys.argv[2])
if not dest.exists():
    dest.write_text(json.dumps(load_config(sys.argv[1]), indent=2))
PY

splits=("0 1" "2 3" "4 5" "6 7" "8 9")
pids=()
for i in "${!splits[@]}"; do
  extra=()
  if [ "$i" -eq 0 ]; then extra+=(--check-flow); fi
  CUDA_VISIBLE_DEVICES=$i bash run.sh collect_rollouts \
    --config "$CONFIG" --run "$RUN" --task-ids ${splits[$i]} "${extra[@]}" \
    > "$ROOT_DIR/collect_g$i.log" 2>&1 &
  pids+=($!)
done
echo "collectors: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done
ls "$ROOT_DIR/buffer"/task*_episode*.pt | wc -l
exit "$fail"
