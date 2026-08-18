"""Backpack velocity fine-tuning for lateral walking and in-place turning.

This task keeps the observations, actions, robot, and legacy rewards compatible
with the trained ``Flat-Backpack-Unitree`` policy.  It deliberately samples
pure lateral and pure-yaw commands, whose exact combinations have effectively
zero probability under the original continuous command sampler.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg

from . import velocity_env_cfg_flat_backpack_unitree as unitree_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Curriculum boundaries in environment steps.  With 24 steps per PPO rollout,
# they correspond to about 3k, 8k, and 15k fine-tuning iterations.
PHASE_STEP_BOUNDARIES = (72_000, 192_000, 360_000)

MODE_STAND = 0
MODE_LEGACY = 1
MODE_CURVE = 2
MODE_LATERAL = 3
MODE_SPIN = 4

# standing, legacy, curve, pure lateral, pure yaw
MODE_PROBABILITIES = (
    (0.05, 0.75, 0.05, 0.10, 0.05),
    (0.05, 0.65, 0.05, 0.15, 0.10),
    (0.05, 0.55, 0.10, 0.15, 0.15),
    (0.05, 0.55, 0.10, 0.15, 0.15),
)

# Minimum and maximum absolute speeds for the specialized modes.  Legacy
# commands retain the completed public curriculum ranges throughout.
LATERAL_SPEED_RANGES = ((0.05, 0.10), (0.05, 0.15), (0.05, 0.20), (0.05, 0.30))
SPIN_SPEED_RANGES = ((0.05, 0.10), (0.05, 0.15), (0.05, 0.20), (0.05, 0.30))


class FineTuneVelocityCommand(UniformVelocityCommand):
    """Preserve legacy locomotion while sampling explicit lateral/spin modes."""

    cfg: FineTuneVelocityCommandCfg

    def __init__(self, cfg: FineTuneVelocityCommandCfg, env):
        super().__init__(cfg, env)
        self.fine_tune_phase = 0
        self.lateral_speed_range = LATERAL_SPEED_RANGES[0]
        self.spin_speed_range = SPIN_SPEED_RANGES[0]
        self.command_mode = torch.full(
            (self.num_envs,), MODE_LEGACY, dtype=torch.long, device=self.device
        )

    @staticmethod
    def _uniform(count: int, value_range: tuple[float, float], device: str) -> torch.Tensor:
        return torch.empty(count, device=device).uniform_(*value_range)

    @staticmethod
    def _signed_uniform(count: int, value_range: tuple[float, float], device: str) -> torch.Tensor:
        magnitude = torch.empty(count, device=device).uniform_(*value_range)
        sign = torch.where(
            torch.rand(count, device=device) < 0.5,
            -torch.ones(count, device=device),
            torch.ones(count, device=device),
        )
        return sign * magnitude

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
        lateral_vy = self._signed_uniform(count, self.lateral_speed_range, self.device)
        spin_wz = self._signed_uniform(count, self.spin_speed_range, self.device)

        selector = torch.rand(count, device=self.device)
        edges = torch.tensor(MODE_PROBABILITIES[self.fine_tune_phase], device=self.device).cumsum(0)
        command = torch.zeros((count, 3), device=self.device)
        mode = torch.full((count,), MODE_LEGACY, dtype=torch.long, device=self.device)

        stand = selector < edges[0]
        legacy = (selector >= edges[0]) & (selector < edges[1])
        curve = (selector >= edges[1]) & (selector < edges[2])
        lateral = (selector >= edges[2]) & (selector < edges[3])
        spin = selector >= edges[3]

        command[legacy, 0] = vx[legacy]
        command[legacy, 1] = vy[legacy]
        command[legacy, 2] = wz[legacy]
        command[curve, 0] = vx[curve]
        command[curve, 2] = wz[curve]
        command[lateral, 1] = lateral_vy[lateral]
        command[spin, 2] = spin_wz[spin]

        mode[stand] = MODE_STAND
        mode[curve] = MODE_CURVE
        mode[lateral] = MODE_LATERAL
        mode[spin] = MODE_SPIN
        self.vel_command_b[env_ids] = command
        self.command_mode[env_ids] = mode
        self.is_standing_env[env_ids] = stand


@configclass
class FineTuneVelocityCommandCfg(mdp.UniformLevelVelocityCommandCfg):
    class_type: type = FineTuneVelocityCommand


def fine_tune_phase(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Expand only the specialized command ranges on a conservative schedule."""
    del env_ids
    command_term: FineTuneVelocityCommand = env.command_manager.get_term("base_velocity")
    target_phase = sum(env.common_step_counter >= boundary for boundary in PHASE_STEP_BOUNDARIES)
    if target_phase > command_term.fine_tune_phase:
        command_term.fine_tune_phase += 1
        command_term.lateral_speed_range = LATERAL_SPEED_RANGES[command_term.fine_tune_phase]
        command_term.spin_speed_range = SPIN_SPEED_RANGES[command_term.fine_tune_phase]
        if command_term.fine_tune_phase == 3:
            command_term.cfg.resampling_time_range = (8.0, 10.0)
    return torch.tensor(command_term.fine_tune_phase, device=env.device, dtype=torch.float)


def _mode_mask(env: ManagerBasedRLEnv, command_name: str, mode: int) -> torch.Tensor:
    command_term: FineTuneVelocityCommand = env.command_manager.get_term(command_name)
    return (command_term.command_mode == mode).float()


def lateral_velocity_error_l1(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Non-saturating lateral tracking error, active only for pure lateral commands."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.abs(command[:, 1] - asset.data.root_lin_vel_b[:, 1])
    return error * _mode_mask(env, command_name, MODE_LATERAL)


def lateral_drift_l1(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize forward and yaw drift during a pure lateral command."""
    asset: Articulation = env.scene[asset_cfg.name]
    drift = torch.abs(asset.data.root_lin_vel_b[:, 0]) + torch.abs(asset.data.root_ang_vel_b[:, 2])
    return drift * _mode_mask(env, command_name, MODE_LATERAL)


def spin_velocity_error_l1(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Non-saturating yaw tracking error, active only for pure-yaw commands."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.abs(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return error * _mode_mask(env, command_name, MODE_SPIN)


def spin_linear_drift_l1(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Keep the base near its original position during an in-place turn."""
    asset: Articulation = env.scene[asset_cfg.name]
    drift = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    return drift * _mode_mask(env, command_name, MODE_SPIN)


def double_support_during_spin(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
) -> torch.Tensor:
    """Gently discourage a planted double-support solution during pure yaw."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    in_contact = torch.linalg.norm(forces, dim=-1) > force_threshold
    double_support = torch.all(in_contact, dim=1).float()
    return double_support * _mode_mask(env, command_name, MODE_SPIN)


def both_feet_airborne(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Prevent the double-support term from being solved by hopping."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    in_contact = torch.linalg.norm(forces, dim=-1) > 1.0
    return (~torch.any(in_contact, dim=1)).float()


def gait_without_spin(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    command_name: str,
) -> torch.Tensor:
    """Keep the learned walking gait but do not impose it on pivot turns."""
    reward = mdp.feet_gait(env, period, offset, sensor_cfg, threshold, command_name)
    return reward * (1.0 - _mode_mask(env, command_name, MODE_SPIN))


def feet_slide_reduced_during_spin(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    spin_scale: float,
) -> torch.Tensor:
    """Allow limited pivot motion without removing the contact-slip penalty."""
    penalty = mdp.feet_slide(env, sensor_cfg=sensor_cfg, asset_cfg=asset_cfg)
    scale = 1.0 - (1.0 - spin_scale) * _mode_mask(env, command_name, MODE_SPIN)
    return penalty * scale


@configclass
class CommandsCfg(unitree_cfg.CommandsCfg):
    base_velocity = FineTuneVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        # Completed legacy curriculum ranges.  These remain fixed; only the
        # pure-lateral and pure-spin ranges above are expanded.
        ranges=FineTuneVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.2, 0.2),
        ),
        limit_ranges=FineTuneVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.2, 0.2),
        ),
    )


@configclass
class RewardsCfg(unitree_cfg.RewardsCfg):
    lateral_velocity_error = RewTerm(
        func=lateral_velocity_error_l1,
        weight=-0.5,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    lateral_drift = RewTerm(
        func=lateral_drift_l1,
        weight=-0.2,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    spin_velocity_error = RewTerm(
        func=spin_velocity_error_l1,
        weight=-0.5,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    spin_linear_drift = RewTerm(
        func=spin_linear_drift_l1,
        weight=-0.2,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    spin_double_support = RewTerm(
        func=double_support_during_spin,
        weight=-0.2,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "force_threshold": 1.0,
        },
    )
    gait = RewTerm(
        func=gait_without_spin,
        weight=1.0,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_slide = RewTerm(
        func=feet_slide_reduced_during_spin,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "command_name": "base_velocity",
            "spin_scale": 0.5,
        },
    )
    airborne = RewTerm(
        func=both_feet_airborne,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"],
            )
        },
    )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    fine_tune_phase = CurrTerm(func=fine_tune_phase)


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


@configclass
class FineTunePPORunnerCfg(BasePPORunnerCfg):
    """Low-update PPO settings for fine-tuning ``model_28000.pt``."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 20_000
        self.save_interval = 100
        self.experiment_name = "unitree_g1_29dof_velocity_flat_backpack_260818"
        self.run_name = "from_pretrained_vel_cur_model_28000"
        self.load_checkpoint = "/workspace/unitree_rl_lab/logs/rsl_rl/pretrained_vel_cur/model_28000.pt"
        self.algorithm.learning_rate = 2.0e-4
        self.algorithm.entropy_coef = 0.005
        self.algorithm.clip_param = 0.15
        self.algorithm.num_learning_epochs = 3

