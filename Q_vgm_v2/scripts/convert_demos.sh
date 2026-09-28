#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=../RLinf/.venv-openpi-robotwin/bin/python
OUT=Q_vgm_v2/artifacts/v2_expert
mkdir -p "$OUT/buffer"
"$PY" - <<'PY'
import json
from pathlib import Path
from qvgm.config import load_config
cfg=load_config('Q_vgm_v2/configs/import_demos.yaml')
p=Path(cfg['paths']['demo_buffer']) / 'config.json'
if p.exists() and json.loads(p.read_text()) != cfg:
    raise ValueError('Demo conversion config differs; refuse resume')
p.write_text(json.dumps(cfg, indent=2))
PY
CUDA_VISIBLE_DEVICES=0 "$PY" -u Q_vgm_v2/scripts/import_libero_demos.py --task-ids 0 1 2 3 4 > "$OUT/gpu0.log" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=1 "$PY" -u Q_vgm_v2/scripts/import_libero_demos.py --task-ids 5 6 7 8 9 > "$OUT/gpu1.log" 2>&1 &
p1=$!
failed=0
wait "$p0" || failed=1
wait "$p1" || failed=1
echo "$failed" > "$OUT/exit_code.txt"
"$PY" Q_vgm_v2/scripts/audit_dataset.py > "$OUT/audit.log"
exit "$failed"
