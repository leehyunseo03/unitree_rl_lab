"""Flat-backpack task variant that learns explicit walk-to-stand behavior.

The physical backpack, observations, actions, domain randomization, and base
locomotion rewards are inherited from ``velocity_env_cfg_flat_backpack``.
This variant increases zero-command sampling, resamples commands within an
episode, and adds command-gated penalties that make stepping in place more
expensive than remaining motionless.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import velocity_env_cfg_flat as flat_cfg
from . import velocity_env_cfg_flat_backpack as backpack_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


STANDING_COMMAND_THRESHOLD = 0.08
STANDING_ENV_FRACTION = 0.20
COMMAND_RESAMPLING_TIME_RANGE = (3.0, 5.0)
EPISODE_LENGTH_S = 12.0

FEET_CFG = SceneEntityCfg("robot", body_names=".*ankle_roll.*")


def _torch(tensor) -> torch.Tensor:
    """Return a torch view for both Tensor and Isaac Lab ProxyArray APIs."""
    return tensor.torch if hasattr(tensor, "torch") else tensor


def stationary_motion_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = FEET_CFG,
    command_threshold: float = STANDING_COMMAND_THRESHOLD,
) -> torch.Tensor:
    """Penalize joint, foot, and root motion only for near-zero commands.

    Velocity tracking alone cannot distinguish a motionless stance from
    stepping in place because both can have nearly zero average root velocity.
    Foot and joint motion therefore provide the direct anti-stepping signal.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    is_standing = (torch.linalg.norm(command, dim=1) < command_threshold).to(command.dtype)

    joint_vel = _torch(asset.data.joint_vel)
    joint_motion = torch.mean(torch.square(joint_vel), dim=1)

    body_lin_vel_w = (
        asset.data.body_link_lin_vel_w
        if hasattr(asset.data, "body_link_lin_vel_w")
        else asset.data.body_lin_vel_w
    )
    foot_lin_vel_w = _torch(body_lin_vel_w)[:, asset_cfg.body_ids, :]
    foot_motion = torch.mean(torch.sum(torch.square(foot_lin_vel_w), dim=-1), dim=1)

    root_lin_vel_b = _torch(asset.data.root_lin_vel_b)
    root_ang_vel_b = _torch(asset.data.root_ang_vel_b)
    root_motion = (
        torch.sum(torch.square(root_lin_vel_b[:, :2]), dim=1)
        + 0.5 * torch.square(root_ang_vel_b[:, 2])
    )

    penalty = 0.2 * joint_motion + foot_motion + 2.0 * root_motion
    return penalty * is_standing


@configclass
class CommandsCfg(backpack_cfg.CommandsCfg):
    """Commands containing frequent stand samples and walk/stand transitions."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=COMMAND_RESAMPLING_TIME_RANGE,
        rel_standing_envs=STANDING_ENV_FRACTION,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.6),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.5, 0.5),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.6),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.5, 0.5),
        ),
    )


@configclass
class RewardsCfg(flat_cfg.RewardsCfg):
    """Base flat rewards plus zero-command standing objectives."""

    stationary_motion = RewTerm(
        func=stationary_motion_penalty,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": FEET_CFG,
            "command_threshold": STANDING_COMMAND_THRESHOLD,
        },
    )
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-0.1,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    feet_contact_standing = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.2,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )


@configclass
class RobotEnvCfg(backpack_cfg.RobotEnvCfg):
    """Training configuration for stable standing and walk-to-stand transitions."""

    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = EPISODE_LENGTH_S


@configclass
class RobotPlayEnvCfg(backpack_cfg.RobotPlayEnvCfg):
    """Deterministic play configuration matching the stand training task."""

    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = EPISODE_LENGTH_S
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
