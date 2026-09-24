"""Configuration and process-local dependency setup; no RLinf imports."""

import os
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs/libero_spatial.yaml"

    def read_config(source, parents=()):
        source = source.resolve()
        if source in parents:
            raise ValueError("Cyclic config inheritance")
        data = yaml.safe_load(source.read_text())
        parent = data.pop("extends", None)
        if parent is None:
            return data

        def merge(base, override):
            result = dict(base)
            for key, value in override.items():
                result[key] = (
                    merge(result[key], value)
                    if isinstance(value, dict) and isinstance(result.get(key), dict)
                    else value
                )
            return result

        return merge(read_config(source.parent / parent, (*parents, source)), data)

    cfg = read_config(path)
    for key, value in cfg["paths"].items():
        # Keep the venv executable path: resolving its symlink loses the venv.
        cfg["paths"][key] = os.path.abspath(ROOT / Path(value).expanduser())
    m, e = cfg["model"], cfg["env"]
    if not 0 < e["action_chunk"] <= m["action_horizon"]:
        raise ValueError("Invalid execution chunk length")
    if m["denoising_steps"] <= 0 or e["action_dim"] != 7:
        raise ValueError("Invalid denoising steps or LIBERO action dimension")
    if cfg["evaluation"]["episodes_per_task"] <= 0:
        raise ValueError("episodes_per_task must be positive")
    if not e["task_ids"] or len(set(e["task_ids"])) != len(e["task_ids"]):
        raise ValueError("task_ids must be nonempty and unique")
    return cfg


def setup_runtime(cfg):
    """Set environment before importing JAX, MuJoCo, LIBERO or OpenPI."""
    paths = cfg["paths"]
    runtime = Path(paths["artifacts"]) / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["MUJOCO_GL"] = cfg["runtime"]["mujoco_gl"]
    os.environ["PYOPENGL_PLATFORM"] = cfg["runtime"]["mujoco_gl"]
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["OMP_NUM_THREADS"] = str(cfg["runtime"]["cpu_threads"])
    os.environ["NUMBA_CACHE_DIR"] = str(runtime / "numba")
    os.environ["XDG_CACHE_HOME"] = str(runtime / "cache")
    # OpenPI's tokenizer has a fixed gs:// URL. Seed its local cache from YAML.
    cache = runtime / "openpi"
    target = cache / "big_vision/paligemma_tokenizer.model"
    target.parent.mkdir(parents=True, exist_ok=True)
    source = Path(paths["tokenizer"])
    if not source.is_file():
        raise FileNotFoundError(source)
    if not target.exists() or source.read_bytes() != target.read_bytes():
        shutil.copyfile(source, target)
    os.environ["OPENPI_DATA_HOME"] = str(cache)
    libero_cfg = runtime / "libero"
    libero_cfg.mkdir(exist_ok=True)
    root = Path(paths["libero_root"])
    data = dict(
        benchmark_root=str(root),
        bddl_files=str(root / "bddl_files"),
        init_states=str(root / "init_files"),
        assets=paths["libero_assets"],
        datasets=str(root.parent / "datasets"),
    )
    (libero_cfg / "config.yaml").write_text(yaml.safe_dump(data))
    os.environ["LIBERO_CONFIG_PATH"] = str(libero_cfg)
    return runtime
