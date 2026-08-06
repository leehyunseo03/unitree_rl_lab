"""Flat G1 velocity task with only the physical backpack added.

All learning settings are inherited from :mod:`velocity_env_cfg_flat`.  This
module only adds the backpack collision geometry and folds its fixed mass,
center of mass, and cuboid inertia into ``torso_link``.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import velocity_env_cfg_flat as base_cfg


# Keep the measured geometry, placement, and component masses from backpack.py.
BACKPACK_BACK_SURFACE_X = -0.070
BACKPACK_CENTER_Y = 0.0
BACKPACK_CENTER_Z = 0.13

BACKPACK_PLATE_THICKNESS = 0.003
BACKPACK_PLATE_WIDTH = 0.135
BACKPACK_PLATE_HEIGHT = 0.24
BACKPACK_COMPONENT_GAP = 0.01
BACKPACK_CONVERTER_WIDTH = 0.05
BACKPACK_CONVERTER_LEFT_INSET = 0.03

BACKPACK_PLATE_SIZE = (BACKPACK_PLATE_THICKNESS, BACKPACK_PLATE_WIDTH, BACKPACK_PLATE_HEIGHT)
BACKPACK_CONVERTER_SIZE = (0.05, BACKPACK_CONVERTER_WIDTH, 0.05)
BACKPACK_JETSON_ORIN_SIZE = (0.05, 0.11, 0.11)
BACKPACK_BATTERY_SIZE = (0.05, 0.11, 0.06)

BACKPACK_PLATE_MASS = 0.170
BACKPACK_CONVERTER_MASS = 0.100
BACKPACK_JETSON_ORIN_MASS = 0.611
BACKPACK_BATTERY_MASS = 0.440

# G1 torso local +Y points left.
BACKPACK_CONVERTER_CENTER_Y = (
    BACKPACK_CENTER_Y
    + 0.5 * BACKPACK_PLATE_WIDTH
    - BACKPACK_CONVERTER_LEFT_INSET
    - 0.5 * BACKPACK_CONVERTER_WIDTH
)

BACKPACK_CONVERTER_Z_OFFSET = 0.5 * BACKPACK_PLATE_HEIGHT - 0.5 * BACKPACK_CONVERTER_SIZE[2]
BACKPACK_JETSON_ORIN_Z_OFFSET = (
    BACKPACK_CONVERTER_Z_OFFSET
    - 0.5 * BACKPACK_CONVERTER_SIZE[2]
    - BACKPACK_COMPONENT_GAP
    - 0.5 * BACKPACK_JETSON_ORIN_SIZE[2]
)
BACKPACK_BATTERY_Z_OFFSET = (
    BACKPACK_JETSON_ORIN_Z_OFFSET
    - 0.5 * BACKPACK_JETSON_ORIN_SIZE[2]
    - BACKPACK_COMPONENT_GAP
    - 0.5 * BACKPACK_BATTERY_SIZE[2]
)


def _plate_local_pos() -> tuple[float, float, float]:
    return (
        BACKPACK_BACK_SURFACE_X - 0.5 * BACKPACK_PLATE_THICKNESS,
        BACKPACK_CENTER_Y,
        BACKPACK_CENTER_Z,
    )


def _component_local_pos(
    size: tuple[float, float, float], z_offset: float, center_y: float = BACKPACK_CENTER_Y
) -> tuple[float, float, float]:
    return (
        BACKPACK_BACK_SURFACE_X - BACKPACK_PLATE_THICKNESS - 0.5 * size[0],
        center_y,
        BACKPACK_CENTER_Z + z_offset,
    )


BACKPACK_PLATE_POS = _plate_local_pos()
BACKPACK_CONVERTER_POS = _component_local_pos(
    BACKPACK_CONVERTER_SIZE, BACKPACK_CONVERTER_Z_OFFSET, BACKPACK_CONVERTER_CENTER_Y
)
BACKPACK_JETSON_ORIN_POS = _component_local_pos(BACKPACK_JETSON_ORIN_SIZE, BACKPACK_JETSON_ORIN_Z_OFFSET)
BACKPACK_BATTERY_POS = _component_local_pos(BACKPACK_BATTERY_SIZE, BACKPACK_BATTERY_Z_OFFSET)

BACKPACK_PARTS = (
    {"name": "plate", "pos": BACKPACK_PLATE_POS, "size": BACKPACK_PLATE_SIZE, "mass": BACKPACK_PLATE_MASS},
    {
        "name": "converter",
        "pos": BACKPACK_CONVERTER_POS,
        "size": BACKPACK_CONVERTER_SIZE,
        "mass": BACKPACK_CONVERTER_MASS,
    },
    {
        "name": "jetson_orin",
        "pos": BACKPACK_JETSON_ORIN_POS,
        "size": BACKPACK_JETSON_ORIN_SIZE,
        "mass": BACKPACK_JETSON_ORIN_MASS,
    },
    {"name": "battery", "pos": BACKPACK_BATTERY_POS, "size": BACKPACK_BATTERY_SIZE, "mass": BACKPACK_BATTERY_MASS},
)

BACKPACK_TOTAL_MASS = sum(float(part["mass"]) for part in BACKPACK_PARTS)
BACKPACK_LOCAL_COM = tuple(
    sum(float(part["pos"][axis]) * float(part["mass"]) for part in BACKPACK_PARTS) / BACKPACK_TOTAL_MASS
    for axis in range(3)
)


def _selected_body_id(asset: Articulation, asset_cfg: SceneEntityCfg) -> int:
    body_ids = asset_cfg.body_ids
    if body_ids == slice(None):
        resolved_ids = list(range(asset.num_bodies))
    elif isinstance(body_ids, int):
        resolved_ids = [body_ids]
    else:
        resolved_ids = list(body_ids)
    if len(resolved_ids) != 1:
        raise ValueError("The backpack payload must target exactly one rigid body.")
    return int(resolved_ids[0])


def _combine_backpack_dynamics(
    masses: torch.Tensor,
    coms: torch.Tensor,
    inertias: torch.Tensor,
    env_ids: torch.Tensor,
    body_id: int,
) -> None:
    """Combine the torso and backpack using cuboid inertia and the parallel-axis theorem."""
    base_mass = masses[env_ids, body_id]
    base_com = coms[env_ids, body_id, :3]
    base_inertia = inertias[env_ids, body_id].reshape(-1, 3, 3)

    dtype = masses.dtype
    device = masses.device
    sample_count = len(env_ids)
    eye = torch.eye(3, dtype=dtype, device=device).expand(sample_count, 3, 3)

    payload_first_moment = torch.zeros((sample_count, 3), dtype=dtype, device=device)
    parsed_parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for part in BACKPACK_PARTS:
        mass = torch.full((sample_count,), float(part["mass"]), dtype=dtype, device=device)
        pos = torch.tensor(part["pos"], dtype=dtype, device=device).expand(sample_count, 3)
        size = torch.tensor(part["size"], dtype=dtype, device=device)
        payload_first_moment += mass[:, None] * pos
        parsed_parts.append((mass, pos, size))

    total_mass = base_mass + BACKPACK_TOTAL_MASS
    combined_com = (base_mass[:, None] * base_com + payload_first_moment) / total_mass[:, None]

    # Shift the original torso inertia from its old COM to the new combined COM.
    combined_inertia = base_inertia.clone()
    base_delta = base_com - combined_com
    combined_inertia += base_mass[:, None, None] * (
        torch.sum(base_delta * base_delta, dim=1)[:, None, None] * eye
        - base_delta[:, :, None] * base_delta[:, None, :]
    )

    # Add each rectangular component's own inertia plus its parallel-axis term.
    for mass, pos, size in parsed_parts:
        sx, sy, sz = size
        part_inertia = torch.zeros((sample_count, 3, 3), dtype=dtype, device=device)
        part_inertia[:, 0, 0] = mass * (sy * sy + sz * sz) / 12.0
        part_inertia[:, 1, 1] = mass * (sx * sx + sz * sz) / 12.0
        part_inertia[:, 2, 2] = mass * (sx * sx + sy * sy) / 12.0
        delta = pos - combined_com
        combined_inertia += part_inertia + mass[:, None, None] * (
            torch.sum(delta * delta, dim=1)[:, None, None] * eye - delta[:, :, None] * delta[:, None, :]
        )

    masses[env_ids, body_id] = total_mass
    coms[env_ids, body_id, :3] = combined_com
    inertias[env_ids, body_id] = combined_inertia.reshape(sample_count, 9)


def apply_fixed_backpack_payload(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Add the fixed 1.321 kg backpack to torso mass, COM, and inertia once at startup."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = _selected_body_id(asset, asset_cfg)

    # Current Isaac Lab tensor API.
    body_mass = getattr(asset.data, "body_mass", None)
    if body_mass is not None and hasattr(body_mass, "torch") and hasattr(asset, "set_masses_index"):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, dtype=torch.int32, device=asset.device)
        else:
            env_ids = env_ids.to(device=asset.device, dtype=torch.int32)

        masses = asset.data.body_mass.torch.clone()
        coms = asset.data.body_com_pose_b.torch.clone()
        inertias = asset.data.body_inertia.torch.clone()
        _combine_backpack_dynamics(masses, coms, inertias, env_ids, body_id)

        body_ids = torch.tensor([body_id], dtype=torch.int32, device=asset.device)
        asset.set_masses_index(
            masses=masses[env_ids[:, None], body_ids], body_ids=body_ids, env_ids=env_ids
        )
        asset.set_coms_index(coms=coms[env_ids[:, None], body_ids], body_ids=body_ids, env_ids=env_ids)
        asset.set_inertias_index(
            inertias=inertias[env_ids[:, None], body_ids], body_ids=body_ids, env_ids=env_ids
        )
        return

    # Isaac Lab 2.x / direct PhysX-view fallback used by the original flat task.
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, dtype=torch.int32, device="cpu")
    else:
        env_ids = env_ids.to(device="cpu", dtype=torch.int32)
    masses = asset.root_physx_view.get_masses().clone()
    coms = asset.root_physx_view.get_coms().clone()
    inertias = asset.root_physx_view.get_inertias().clone()
    _combine_backpack_dynamics(masses, coms, inertias, env_ids, body_id)
    asset.root_physx_view.set_masses(masses, env_ids)
    asset.root_physx_view.set_coms(coms, env_ids)
    asset.root_physx_view.set_inertias(inertias, env_ids)


@configclass
class RobotSceneCfg(base_cfg.RobotSceneCfg):
    """The original flat scene plus backpack collision geometry."""

    # These are collider shapes on torso_link, not separate rigid bodies.  Their
    # physical mass is injected exactly once by apply_fixed_backpack_payload.
    backpack_plate = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/backpack_plate",
        init_state=AssetBaseCfg.InitialStateCfg(pos=BACKPACK_PLATE_POS),
        spawn=sim_utils.CuboidCfg(
            size=BACKPACK_PLATE_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02, 0.025, 0.03), roughness=0.9),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )
    backpack_converter = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/backpack_converter",
        init_state=AssetBaseCfg.InitialStateCfg(pos=BACKPACK_CONVERTER_POS),
        spawn=sim_utils.CuboidCfg(
            size=BACKPACK_CONVERTER_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.20, 0.22), roughness=0.75),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )
    backpack_jetson_orin = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/backpack_jetson_orin",
        init_state=AssetBaseCfg.InitialStateCfg(pos=BACKPACK_JETSON_ORIN_POS),
        spawn=sim_utils.CuboidCfg(
            size=BACKPACK_JETSON_ORIN_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.18, 0.12), roughness=0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )
    backpack_battery = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/backpack_battery",
        init_state=AssetBaseCfg.InitialStateCfg(pos=BACKPACK_BATTERY_POS),
        spawn=sim_utils.CuboidCfg(
            size=BACKPACK_BATTERY_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.055, 0.06), roughness=0.85),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )


@configclass
class EventCfg(base_cfg.EventCfg):
    """The original flat events plus one fixed backpack payload event."""

    fixed_backpack_payload = EventTerm(
        func=apply_fixed_backpack_payload,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )


@configclass
class RobotEnvCfg(base_cfg.RobotEnvCfg):
    """Original flat training configuration with only the backpack added."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    events: EventCfg = EventCfg()


@configclass
class RobotPlayEnvCfg(base_cfg.RobotPlayEnvCfg):
    """Original flat play configuration with the same fixed backpack."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=32, env_spacing=2.5)
    events: EventCfg = EventCfg()
