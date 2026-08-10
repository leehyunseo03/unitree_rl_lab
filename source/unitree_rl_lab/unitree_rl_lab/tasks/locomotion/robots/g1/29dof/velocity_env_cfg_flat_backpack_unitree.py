"""Official Unitree G1 velocity task with backpack-only environment changes.

Commands, rewards, observations, actions, terminations, terrain curriculum,
velocity curriculum, episode length, and simulation timing match the public
``Unitree-G1-29dof-Velocity`` configuration.  The only intended differences
are the physical backpack scene and the backpack/floor domain randomization
inherited from :mod:`velocity_env_cfg_flat_backpack`.

The local repository changed ``mdp.foot_clearance_reward`` after forking the
public project.  This module therefore carries the public reward formula
locally so that other tasks keep their existing behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import velocity_env_cfg_flat as flat_cfg
from . import velocity_env_cfg_flat_backpack as backpack_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


EPISODE_LENGTH_S = 20.0


def _torch(tensor) -> torch.Tensor:
    """Return a torch view for both Tensor and Isaac Lab ProxyArray APIs."""
    return tensor.torch if hasattr(tensor, "torch") else tensor


def unitree_foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Public Unitree foot-clearance reward with API-version compatibility."""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_link_pos_w if hasattr(asset.data, "body_link_pos_w") else asset.data.body_pos_w
    body_lin_vel_w = (
        asset.data.body_link_lin_vel_w
        if hasattr(asset.data, "body_link_lin_vel_w")
        else asset.data.body_lin_vel_w
    )
    foot_pos_w = _torch(body_pos_w)[:, asset_cfg.body_ids, :]
    foot_lin_vel_w = _torch(body_lin_vel_w)[:, asset_cfg.body_ids, :]

    foot_z_target_error = torch.square(foot_pos_w[:, :, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.linalg.norm(foot_lin_vel_w[:, :, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


@configclass
class CommandsCfg(backpack_cfg.CommandsCfg):
    """Public Unitree command ranges and resampling schedule."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.1, 0.1),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.2, 0.2),
        ),
    )


@configclass
class RewardsCfg(flat_cfg.RewardsCfg):
    """Public Unitree G1 rewards, including its original foot reward."""

    feet_clearance = RewTerm(
        func=unitree_foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )


@configclass
class CurriculumCfg:
    """Public Unitree terrain and linear-velocity curricula."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(backpack_cfg.RobotEnvCfg):
    """Unitree-equivalent training configuration with backpack dynamics."""

    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = EPISODE_LENGTH_S


@configclass
class RobotPlayEnvCfg(backpack_cfg.RobotPlayEnvCfg):
    """Unitree-equivalent play configuration with backpack dynamics."""

    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = EPISODE_LENGTH_S
