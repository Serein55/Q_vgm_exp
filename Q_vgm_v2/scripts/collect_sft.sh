#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=../RLinf/.venv-openpi-robotwin/bin/python
CFG=Q_vgm_v2/configs/collect_sft.yaml
OUT=Q_vgm_v2/artifacts/v2_sft_500
mkdir -p "$OUT/buffer"
# Single writer creates the manifest before workers start.
"$PY" - "$CFG" "$OUT/buffer/config.json" <<'PY'
import json, sys
from pathlib import Path
from qvgm.config import load_config
p = Path(sys.argv[2])
cfg = load_config(sys.argv[1])
if p.exists():
    if json.loads(p.read_text()) != cfg:
        raise ValueError('Collection config changed; refuse resume')
else:
    p.write_text(json.dumps(cfg, indent=2))
PY
CUDA_VISIBLE_DEVICES=2 "$PY" -u scripts/collect_rollouts.py --config "$CFG" --run v2_sft_500 --task-ids 0 1 2 3 4 > "$OUT/gpu2.log" 2>&1 &
p2=$!
CUDA_VISIBLE_DEVICES=3 "$PY" -u scripts/collect_rollouts.py --config "$CFG" --run v2_sft_500 --task-ids 5 6 7 8 9 > "$OUT/gpu3.log" 2>&1 &
p3=$!
failed=0
wait "$p2" || failed=1
wait "$p3" || failed=1
"$PY" Q_vgm_v2/scripts/summarize_collection.py "$OUT/buffer" > "$OUT/summary.log"
echo "$failed" > "$OUT/exit_code.txt"
exit "$failed"
