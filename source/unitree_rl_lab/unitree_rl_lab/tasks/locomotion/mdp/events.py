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


def apply_fixed_payload_to_rigid_body(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    payload_parts: tuple[dict[str, object], ...],
):
    """Merge fixed payload parts into one body's mass, CoM, and inertia."""
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device, dtype=torch.int32)
    else:
        env_ids = env_ids.to(asset.device, dtype=torch.int32)

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int32, device=asset.device)
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int32, device=asset.device)
    if len(body_ids) != 1:
        raise ValueError("apply_fixed_payload_to_rigid_body expects exactly one target body.")

    body_id = body_ids[0]
    masses = asset.data.body_mass.torch.clone()
    coms = asset.data.body_com_pose_b.torch.clone()
    inertias = asset.data.body_inertia.torch.clone()

    base_mass = masses[env_ids, body_id]
    base_com = coms[env_ids, body_id, :3]
    base_inertia = inertias[env_ids, body_id].reshape(-1, 3, 3)

    payload_mass = 0.0
    payload_mass_pos = torch.zeros((3,), device=asset.device, dtype=torch.float32)
    parsed_parts = []
    for part in payload_parts:
        mass = float(part["mass"])
        pos = torch.tensor(part["pos"], device=asset.device, dtype=torch.float32)
        size = torch.tensor(part["size"], device=asset.device, dtype=torch.float32)
        payload_mass += mass
        payload_mass_pos += mass * pos
        parsed_parts.append((mass, pos, size))

    total_mass = base_mass + payload_mass
    combined_com = (base_mass[:, None] * base_com + payload_mass_pos[None, :]) / total_mass[:, None]

    eye = torch.eye(3, device=asset.device, dtype=torch.float32).expand(len(env_ids), 3, 3)
    combined_inertia = base_inertia.clone()

    base_delta = base_com - combined_com
    combined_inertia += base_mass[:, None, None] * (
        torch.sum(base_delta * base_delta, dim=1)[:, None, None] * eye
        - base_delta[:, :, None] * base_delta[:, None, :]
    )

    for mass, pos, size in parsed_parts:
        sx, sy, sz = size
        part_inertia = torch.diag(
            torch.stack(
                (
                    mass * (sy * sy + sz * sz) / 12.0,
                    mass * (sx * sx + sz * sz) / 12.0,
                    mass * (sx * sx + sy * sy) / 12.0,
                )
            )
        )
        delta = pos[None, :] - combined_com
        combined_inertia += part_inertia[None, :, :] + mass * (
            torch.sum(delta * delta, dim=1)[:, None, None] * eye - delta[:, :, None] * delta[:, None, :]
        )

    masses[env_ids, body_id] = total_mass
    coms[env_ids, body_id, :3] = combined_com
    inertias[env_ids, body_id] = combined_inertia.reshape(len(env_ids), 9)

    asset.set_masses_index(masses=masses[env_ids[:, None], body_ids], body_ids=body_ids, env_ids=env_ids)
    asset.set_coms_index(coms=coms[env_ids[:, None], body_ids], body_ids=body_ids, env_ids=env_ids)
    asset.set_inertias_index(inertias=inertias[env_ids[:, None], body_ids], body_ids=body_ids, env_ids=env_ids)


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
