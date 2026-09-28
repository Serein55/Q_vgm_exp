"""Download the public source used by LIBERO's own downloader at a pinned revision."""

import json
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parents[1] / "artifacts/expert_source"
root.mkdir(parents=True, exist_ok=True)
source = dict(
    repo_id="yifengzhu-hf/LIBERO-datasets",
    repo_type="dataset",
    revision="f13aa24a3da8c43c7225569f28c562979fa0e35a",
)
(root / "source.json").write_text(json.dumps(source, indent=2))
snapshot_download(
    **source, allow_patterns="libero_spatial/*", local_dir=root, token=False, max_workers=2
)
