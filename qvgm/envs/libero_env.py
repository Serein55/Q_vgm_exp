"""LIBERO observation contract matching the released OpenPI checkpoint."""

from pathlib import Path

import numpy as np

DUMMY_ACTION = [0.0] * 6 + [-1.0]


def quat_to_axisangle(quat):
    q = np.asarray(quat, dtype=np.float64)
    w = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    if den < 1e-8:
        return np.zeros(3)
    return q[:3] * (2.0 * np.arccos(w) / den)


def observation_for_policy(obs, language):
    return {
        "observation/image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
        "observation/wrist_image": np.ascontiguousarray(
            obs["robot0_eye_in_hand_image"][::-1, ::-1]
        ),
        "observation/state": np.concatenate(
            [
                obs["robot0_eef_pos"],
                quat_to_axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ]
        ).astype(np.float32),
        "prompt": str(language),
    }


def make_env(cfg, task, seed):
    from libero.libero.envs import OffScreenRenderEnv

    path = Path(cfg["paths"]["libero_root"]) / "bddl_files" / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(path),
        camera_heights=cfg["env"]["resolution"],
        camera_widths=cfg["env"]["resolution"],
    )
    env.seed(seed)
    return env
