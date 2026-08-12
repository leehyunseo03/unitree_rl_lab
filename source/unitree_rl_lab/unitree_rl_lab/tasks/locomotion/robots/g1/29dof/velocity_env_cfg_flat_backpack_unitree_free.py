"""Unitree flat-backpack task with a free-locomotion command curriculum.

The curriculum deliberately samples command modes that have measure zero in a
plain continuous sampler (most importantly pure yaw).  Training starts with
straight and gentle curved walking, then introduces lateral motion and
in-place turns, and finally mixes all three velocity axes while shortening the
time between command changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import velocity_env_cfg_flat_backpack_unitree as unitree_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class FreeLocomotionVelocityCommand(UniformVelocityCommand):
    """Velocity sampler with explicit standing, translation, and turning modes."""

    cfg: FreeLocomotionVelocityCommandCfg

    def __init__(self, cfg: FreeLocomotionVelocityCommandCfg, env):
        super().__init__(cfg, env)
        self.free_locomotion_level = 0

    @staticmethod
    def _uniform(count: int, value_range: tuple[float, float], device: str) -> torch.Tensor:
        return torch.empty(count, device=device).uniform_(*value_range)

    def _resample_command(self, env_ids: Sequence[int]):
        # Let the upstream implementation maintain heading-related state, then
        # replace its continuous samples with deliberate locomotion modes.
        super()._resample_command(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        count = len(env_ids)
        if count == 0:
            return

        ranges = self.cfg.ranges
        vx = self._uniform(count, ranges.lin_vel_x, self.device)
        vy = self._uniform(count, ranges.lin_vel_y, self.device)
        wz = self._uniform(count, ranges.ang_vel_z, self.device)
        selector = torch.rand(count, device=self.device)
        command = torch.zeros((count, 3), device=self.device)

        # Level 0: standing 10%, straight 70%, gentle curves 20%.
        # Level 1: standing 10%, straight 35%, curves 25%, lateral 15%, spin 15%.
        # Level 2: standing 10%, straight 20%, curves 20%, lateral 15%, spin 15%, mixed 20%.
        if self.free_locomotion_level == 0:
            straight = (selector >= 0.10) & (selector < 0.80)
            curve = selector >= 0.80
            command[straight, 0] = vx[straight]
            command[curve, 0] = vx[curve]
            command[curve, 2] = 0.5 * wz[curve]
        elif self.free_locomotion_level == 1:
            straight = (selector >= 0.10) & (selector < 0.45)
            curve = (selector >= 0.45) & (selector < 0.70)
            lateral = (selector >= 0.70) & (selector < 0.85)
            spin = selector >= 0.85
            command[straight, 0] = vx[straight]
            command[curve, 0] = vx[curve]
            command[curve, 2] = wz[curve]
            command[lateral, 1] = vy[lateral]
            command[spin, 2] = wz[spin]
        else:
            straight = (selector >= 0.10) & (selector < 0.30)
            curve = (selector >= 0.30) & (selector < 0.50)
            lateral = (selector >= 0.50) & (selector < 0.65)
            spin = (selector >= 0.65) & (selector < 0.80)
            mixed = selector >= 0.80
            command[straight, 0] = vx[straight]
            command[curve, 0] = vx[curve]
            command[curve, 2] = wz[curve]
            command[lateral, 1] = vy[lateral]
            command[spin, 2] = wz[spin]
            command[mixed, 0] = vx[mixed]
            command[mixed, 1] = vy[mixed]
            command[mixed, 2] = wz[mixed]

        self.vel_command_b[env_ids] = command
        self.is_standing_env[env_ids] = selector < 0.10


@configclass
class FreeLocomotionVelocityCommandCfg(mdp.UniformLevelVelocityCommandCfg):
    """Configuration binding the mode-aware sampler to Isaac Lab."""

    class_type: type = FreeLocomotionVelocityCommand


def free_locomotion_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    linear_reward_name: str = "track_lin_vel_xy",
    angular_reward_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    """Expand command ranges, command modes, and transition frequency by performance."""
    command_term: FreeLocomotionVelocityCommand = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limits = command_term.cfg.limit_ranges

    linear_cfg = env.reward_manager.get_term_cfg(linear_reward_name)
    angular_cfg = env.reward_manager.get_term_cfg(angular_reward_name)
    linear_reward = torch.mean(env.reward_manager._episode_sums[linear_reward_name][env_ids])
    angular_reward = torch.mean(env.reward_manager._episode_sums[angular_reward_name][env_ids])
    linear_reward /= env.max_episode_length_s
    angular_reward /= env.max_episode_length_s

    # Curriculum terms run on reset.  Synchronizing updates to one episode
    # boundary prevents thousands of parallel resets from skipping levels.
    if env.common_step_counter % env.max_episode_length == 0:
        linear_ready = linear_reward > linear_cfg.weight * 0.8
        angular_ready = angular_reward > angular_cfg.weight * 0.8

        if linear_ready:
            delta = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta,
                limits.lin_vel_x[0],
                limits.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta,
                limits.lin_vel_y[0],
                limits.lin_vel_y[1],
            ).tolist()

        if angular_ready:
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device)
                + torch.tensor([-0.1, 0.1], device=env.device),
                limits.ang_vel_z[0],
                limits.ang_vel_z[1],
            ).tolist()

        if linear_ready and angular_ready and command_term.free_locomotion_level < 2:
            command_term.free_locomotion_level += 1
            command_term.cfg.resampling_time_range = (
                (5.0, 8.0) if command_term.free_locomotion_level == 1 else (2.0, 5.0)
            )

    return torch.tensor(command_term.free_locomotion_level, device=env.device, dtype=torch.float)


@configclass
class CommandsCfg(unitree_cfg.CommandsCfg):
    """Staged command ranges with a faster final in-place yaw rate."""

    base_velocity = FreeLocomotionVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=FreeLocomotionVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.1, 0.1),
        ),
        limit_ranges=FreeLocomotionVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.8, 0.8),
        ),
    )


@configclass
class CurriculumCfg:
    """Keep terrain progression and add the unified free-locomotion curriculum."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    free_locomotion_levels = CurrTerm(func=free_locomotion_levels)


@configclass
class RobotEnvCfg(unitree_cfg.RobotEnvCfg):
    """Training configuration for freely changing planar velocity commands."""

    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class RobotPlayEnvCfg(unitree_cfg.RobotPlayEnvCfg):
    """Play configuration matching free-locomotion training."""

    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
