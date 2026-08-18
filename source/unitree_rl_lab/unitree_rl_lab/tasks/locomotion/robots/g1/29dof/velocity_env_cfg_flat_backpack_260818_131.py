"""Backpack fine-tuning variant containing only Unitree issue #131 changes.

Relative to :mod:`velocity_env_cfg_flat_backpack_unitree`, this configuration
only widens the yaw-rate command range and increases the exponential yaw-rate
tracking reward.  All curricula and every other reward remain unchanged.
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import velocity_env_cfg_flat_backpack_unitree as unitree_cfg


@configclass
class CommandsCfg(unitree_cfg.CommandsCfg):
    """Original commands with only issue #131's yaw ranges applied."""

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
            ang_vel_z=(-0.3, 0.3),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class RewardsCfg(unitree_cfg.RewardsCfg):
    """Original rewards with only issue #131's yaw weight applied."""

    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )


@configclass
class RobotEnvCfg(unitree_cfg.RobotEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(unitree_cfg.RobotPlayEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()

