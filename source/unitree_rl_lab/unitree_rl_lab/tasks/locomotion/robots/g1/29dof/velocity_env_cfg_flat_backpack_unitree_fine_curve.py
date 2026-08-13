"""Conservative free-locomotion fine-tuning on top of a trained Unitree policy.

This task is intended to be started with ``--resume`` from a stable backpack
velocity/curve checkpoint.  Most samples keep the original continuous command
distribution.  Pure-yaw and other specialized modes are introduced slowly on
a fixed minimum-step schedule so easy near-zero tracking rewards cannot skip
the curriculum.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import velocity_env_cfg_flat_backpack_unitree as unitree_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# With the default RSL-RL rollout length (24), these boundaries are roughly
# 5k, 12.5k, and 25k fine-tuning iterations.  They are deliberately time based:
# aggregate tracking reward is misleading when most yaw commands are zero.
PHASE_STEP_BOUNDARIES = (120_000, 300_000, 600_000)

MODE_STAND = 0
MODE_LEGACY = 1
MODE_CURVE = 2
MODE_LATERAL = 3
MODE_SPIN = 4


class FineCurveVelocityCommand(UniformVelocityCommand):
    """Preserve legacy commands while gradually adding deliberate turn modes."""

    cfg: FineCurveVelocityCommandCfg

    def __init__(self, cfg: FineCurveVelocityCommandCfg, env):
        super().__init__(cfg, env)
        self.fine_curve_phase = 0
        self.command_mode = torch.full(
            (self.num_envs,), MODE_LEGACY, dtype=torch.long, device=self.device
        )

    @staticmethod
    def _uniform(count: int, value_range: tuple[float, float], device: str) -> torch.Tensor:
        return torch.empty(count, device=device).uniform_(*value_range)

    def _resample_command(self, env_ids: Sequence[int]):
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
        mode = torch.full((count,), MODE_LEGACY, dtype=torch.long, device=self.device)

        # Each row is: standing, legacy continuous, curve, lateral, pure yaw.
        # Phase 0 has no pure yaw and keeps 85% of the successful old task.
        probabilities = (
            (0.05, 0.85, 0.10, 0.00, 0.00),
            (0.05, 0.80, 0.10, 0.00, 0.05),
            (0.05, 0.70, 0.15, 0.00, 0.10),
            (0.05, 0.60, 0.15, 0.05, 0.15),
        )[self.fine_curve_phase]
        edges = torch.tensor(probabilities, device=self.device).cumsum(0)

        stand = selector < edges[0]
        legacy = (selector >= edges[0]) & (selector < edges[1])
        curve = (selector >= edges[1]) & (selector < edges[2])
        lateral = (selector >= edges[2]) & (selector < edges[3])
        spin = selector >= edges[3]

        # Legacy samples exactly retain simultaneous vx/vy/wz commands.
        command[legacy, 0] = vx[legacy]
        command[legacy, 1] = vy[legacy]
        command[legacy, 2] = wz[legacy]
        command[curve, 0] = vx[curve]
        command[curve, 2] = wz[curve]
        command[lateral, 1] = vy[lateral]
        command[spin, 2] = wz[spin]

        mode[stand] = MODE_STAND
        mode[curve] = MODE_CURVE
        mode[lateral] = MODE_LATERAL
        mode[spin] = MODE_SPIN
        self.vel_command_b[env_ids] = command
        self.command_mode[env_ids] = mode
        self.is_standing_env[env_ids] = stand


@configclass
class FineCurveVelocityCommandCfg(mdp.UniformLevelVelocityCommandCfg):
    class_type: type = FineCurveVelocityCommand


def fine_curve_phase(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Advance only after substantial fine-tuning time; never skip a phase."""
    del env_ids
    command_term: FineCurveVelocityCommand = env.command_manager.get_term("base_velocity")
    target_phase = sum(env.common_step_counter >= boundary for boundary in PHASE_STEP_BOUNDARIES)
    if target_phase > command_term.fine_curve_phase:
        command_term.fine_curve_phase += 1
        yaw_limits = (0.2, 0.3, 0.45, 0.6)
        yaw_limit = yaw_limits[command_term.fine_curve_phase]
        command_term.cfg.ranges.ang_vel_z = (-yaw_limit, yaw_limit)
        # Keep long, predictable commands while learning turns.  Only the final
        # phase adds a small amount of transition difficulty.
        if command_term.fine_curve_phase == 3:
            command_term.cfg.resampling_time_range = (8.0, 10.0)
    return torch.tensor(command_term.fine_curve_phase, device=env.device, dtype=torch.float)


def gait_without_spin(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    command_name: str,
) -> torch.Tensor:
    """Do not force the straight-walking gait template during pure-yaw turns."""
    reward = mdp.feet_gait(env, period, offset, sensor_cfg, threshold, command_name)
    command_term: FineCurveVelocityCommand = env.command_manager.get_term(command_name)
    return reward * (command_term.command_mode != MODE_SPIN)


def both_feet_airborne(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize hopping, measured as both feet losing contact simultaneously."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    in_contact = torch.linalg.norm(forces, dim=-1) > 1.0
    return (~torch.any(in_contact, dim=1)).float()


@configclass
class CommandsCfg(unitree_cfg.CommandsCfg):
    base_velocity = FineCurveVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        # Start from the completed public linear curriculum and the existing
        # curve task's yaw limit; do not make the checkpoint relearn walking.
        ranges=FineCurveVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.2, 0.2),
        ),
        limit_ranges=FineCurveVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.6, 0.6),
        ),
    )


@configclass
class RewardsCfg(unitree_cfg.RewardsCfg):
    # Retain the learned gait for normal/curve commands, but do not impose it
    # on pivot turns where it encouraged hopping in the previous free task.
    gait = RewTerm(
        func=gait_without_spin,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    airborne = RewTerm(
        func=both_feet_airborne,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")},
    )
    # The parent already penalizes vertical speed at -2.0.  A modest increase
    # makes velocity-tracking-by-jumping less attractive during fine-tuning.
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-3.0)


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    fine_curve_phase = CurrTerm(func=fine_curve_phase)


@configclass
class RobotEnvCfg(unitree_cfg.RobotEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class RobotPlayEnvCfg(unitree_cfg.RobotPlayEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
