#!/usr/bin/env bash
# Detached coordinator: final audit only after both workers have terminated.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=../RLinf/.venv-openpi-robotwin/bin/python
ROOT=Q_vgm_v2/artifacts
while [[ ! -f "$ROOT/v2_sft_500/exit_code.txt" || ! -f "$ROOT/v2_expert/exit_code.txt" ]]; do
  sleep 60
done
"$PY" Q_vgm_v2/scripts/summarize_collection.py "$ROOT/v2_sft_500/buffer" > "$ROOT/v2_sft_500/summary.log"
"$PY" Q_vgm_v2/scripts/audit_dataset.py > "$ROOT/final_audit.log"
"$PY" - <<'PY'
import json
from pathlib import Path
root=Path('Q_vgm_v2/artifacts')
r=json.loads((root/'dataset_report.json').read_text())
s=json.loads((root/'v2_sft_500/success_once_summary.json').read_text())
text='# v2 数据准备状态\n\n'
text+=f"SFT：{s['episodes']}/500；success_once：{s['success_once']}/{s['episodes']}。\n\n"
text+=f"Expert：{r['sources']['demo']['episodes']}/500（官方原始版本，论文 432 条子集未确认）。\n\n"
text+=f"结构/标签审计通过且两类数据齐全：{r['ready']}；异常数：{len(r['errors'])}。\n\n"
text+='详细记录：`artifacts/dataset_report.json`、`artifacts/combined_manifest.json`。\n'
Path('Q_vgm_v2/数据准备状态.md').write_text(text)
PY
