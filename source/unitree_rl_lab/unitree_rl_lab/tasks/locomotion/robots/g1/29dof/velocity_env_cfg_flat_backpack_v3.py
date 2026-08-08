"""Torso-stability reward variant (v3) for the flat G1 backpack task.

The robot, backpack dynamics, observations, commands, events, terminations,
and curriculum are inherited unchanged from
``velocity_env_cfg_flat_backpack``.  This variant only changes rewards.

Unlike an absolute-yaw penalty, the yaw terms below do not prevent commanded
turning.  They penalize torso yaw motion relative to the command and relative
to the pelvis, while the pitch terms keep the backpack-carrying torso upright
and damp its fore/aft rocking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

try:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
except ImportError:
    # Compatibility with Isaac Lab releases that used the old function name.
    from isaaclab.utils.math import quat_apply, quat_rotate_inverse as quat_apply_inverse

from . import velocity_env_cfg_flat as flat_cfg
from . import velocity_env_cfg_flat_backpack as backpack_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


TORSO_CFG = SceneEntityCfg("robot", body_names="torso_link")


def _torch(tensor):
    """Return a torch view for both Tensor and Isaac Lab ProxyArray APIs."""
    return tensor.torch if hasattr(tensor, "torch") else tensor


def _torso_kinematics(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> tuple[Articulation, torch.Tensor, torch.Tensor]:
    """Return the robot, torso orientation, and torso-frame angular velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    # ``body_link_*`` is the current API; ``body_*`` keeps v3 usable with the
    # older Isaac Lab release supported by this repository as well.
    body_quat_w = (
        asset.data.body_link_quat_w if hasattr(asset.data, "body_link_quat_w") else asset.data.body_quat_w
    )
    body_ang_vel_w = (
        asset.data.body_link_ang_vel_w if hasattr(asset.data, "body_link_ang_vel_w") else asset.data.body_ang_vel_w
    )
    torso_quat_w = _torch(body_quat_w)[:, asset_cfg.body_ids]
    torso_ang_vel_w = _torch(body_ang_vel_w)[:, asset_cfg.body_ids]
    torso_ang_vel_b = quat_apply_inverse(torso_quat_w, torso_ang_vel_w)
    return asset, torso_quat_w, torso_ang_vel_b


def torso_pitch_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = TORSO_CFG,
) -> torch.Tensor:
    """Penalize torso pitch angle using gravity projected into the torso frame."""
    _, torso_quat_w, _ = _torso_kinematics(env, asset_cfg)
    gravity_w = torch.zeros_like(torso_quat_w[..., :3])
    gravity_w[..., 2] = -1.0
    projected_gravity = quat_apply_inverse(torso_quat_w, gravity_w)
    return torch.sum(torch.square(projected_gravity[..., 0]), dim=-1)


def torso_pitch_ang_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = TORSO_CFG,
) -> torch.Tensor:
    """Penalize torso pitch angular velocity (rotation about local Y)."""
    _, _, torso_ang_vel_b = _torso_kinematics(env, asset_cfg)
    return torch.sum(torch.square(torso_ang_vel_b[..., 1]), dim=-1)


def torso_yaw_rate_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = TORSO_CFG,
) -> torch.Tensor:
    """Penalize torso yaw-rate error without suppressing commanded turns."""
    _, _, torso_ang_vel_b = _torso_kinematics(env, asset_cfg)
    target_yaw_rate = env.command_manager.get_command(command_name)[:, 2].unsqueeze(-1)
    return torch.sum(torch.square(torso_ang_vel_b[..., 2] - target_yaw_rate), dim=-1)


def torso_pelvis_yaw_error_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = TORSO_CFG,
) -> torch.Tensor:
    """Penalize torso heading oscillation relative to the pelvis heading."""
    asset, torso_quat_w, _ = _torso_kinematics(env, asset_cfg)
    root_quat_w = (
        asset.data.root_link_quat_w if hasattr(asset.data, "root_link_quat_w") else asset.data.root_quat_w
    )
    pelvis_quat_w = _torch(root_quat_w).unsqueeze(1)

    forward_b = torch.zeros_like(torso_quat_w[..., :3])
    forward_b[..., 0] = 1.0
    torso_forward_w = quat_apply(torso_quat_w, forward_b)
    pelvis_forward_w = quat_apply(pelvis_quat_w, forward_b)

    torso_heading = torch.atan2(torso_forward_w[..., 1], torso_forward_w[..., 0])
    pelvis_heading = torch.atan2(pelvis_forward_w[..., 1], pelvis_forward_w[..., 0])
    heading_error = torch.atan2(
        torch.sin(torso_heading - pelvis_heading),
        torch.cos(torso_heading - pelvis_heading),
    )
    return torch.sum(torch.square(heading_error), dim=-1)


_V1_REWARDS = flat_cfg.RewardsCfg()


@configclass
class RewardsCfg(flat_cfg.RewardsCfg):
    """V1 backpack rewards plus explicit torso pitch/yaw stabilization."""

    # Preserve strong locomotion tracking while improving yaw-rate tracking.
    track_ang_vel_z = _V1_REWARDS.track_ang_vel_z.replace(weight=0.75)

    # Slightly strengthen existing pelvis/trunk smoothness terms.
    base_angular_velocity = _V1_REWARDS.base_angular_velocity.replace(weight=-0.10)
    flat_orientation_l2 = _V1_REWARDS.flat_orientation_l2.replace(weight=-6.0)
    joint_deviation_waists = _V1_REWARDS.joint_deviation_waists.replace(weight=-1.5)
    action_rate = _V1_REWARDS.action_rate.replace(weight=-0.06)

    # Explicit torso terms.  These observe torso_link rather than only the root
    # pelvis, which is important because the backpack is attached to torso_link.
    torso_pitch = RewTerm(func=torso_pitch_l2, weight=-5.0, params={"asset_cfg": TORSO_CFG})
    torso_pitch_rate = RewTerm(func=torso_pitch_ang_vel_l2, weight=-0.15, params={"asset_cfg": TORSO_CFG})
    torso_yaw_rate_error = RewTerm(
        func=torso_yaw_rate_error_l2,
        weight=-0.25,
        params={"command_name": "base_velocity", "asset_cfg": TORSO_CFG},
    )
    torso_pelvis_yaw_error = RewTerm(
        func=torso_pelvis_yaw_error_l2,
        weight=-2.0,
        params={"asset_cfg": TORSO_CFG},
    )


@configclass
class RobotEnvCfg(backpack_cfg.RobotEnvCfg):
    """Flat backpack training configuration using the v3 rewards."""

    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(backpack_cfg.RobotPlayEnvCfg):
    """Play configuration matching :class:`RobotEnvCfg`."""

    rewards: RewardsCfg = RewardsCfg()
