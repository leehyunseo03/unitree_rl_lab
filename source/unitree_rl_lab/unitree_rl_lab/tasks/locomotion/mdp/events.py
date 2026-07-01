from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize rigid-body CoM offsets by adding sampled local-frame offsets."""
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = np.arange(env.scene.num_envs, dtype=np.uint32)
    else:
        env_ids = env_ids.cpu().numpy().astype(np.uint32)

    if asset_cfg.body_ids == slice(None):
        body_ids = np.arange(asset.num_bodies)
    else:
        body_ids = np.asarray(asset_cfg.body_ids, dtype=np.int64)

    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").numpy()

    coms = np.array(asset.root_physx_view.get_coms().numpy(), copy=True)
    for sample_id, env_id in enumerate(env_ids):
        coms[env_id, body_ids, :3] += rand_samples[sample_id]

    try:
        asset.root_physx_view.set_coms(
            wp.array(coms, dtype=wp.float32, device="cpu"),
            wp.array(env_ids, dtype=wp.uint32, device="cpu"),
        )
    except Exception as exc:
        print(f"[WARN] Skipping CoM randomization because PhysX rejected set_coms: {exc}")
