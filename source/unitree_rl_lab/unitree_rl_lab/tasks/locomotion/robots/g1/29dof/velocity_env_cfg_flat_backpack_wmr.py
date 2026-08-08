"""Flat G1 backpack task using the World Model Reconstruction reward set.

The environment, backpack dynamics, commands, observations, actions,
terminations, and curriculum come from ``velocity_env_cfg_flat_backpack``.
Only the rewards are replaced with the WMR rewards documented in
``G1_Locomotion_인수인계.pdf`` and the WMR paper appendix.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import velocity_env_cfg_flat_backpack as backpack_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _torch(tensor):
    """Return a torch view for Tensor and Isaac Lab ProxyArray APIs."""
    return tensor.torch if hasattr(tensor, "torch") else tensor


def feet_force_wmr(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
    max_excess_force: float,
) -> torch.Tensor:
    """Reward capped vertical ground-reaction force above the WMR threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_w = _torch(contact_sensor.data.net_forces_w)[:, sensor_cfg.body_ids]
    vertical_force = torch.clamp(forces_w[..., 2], min=0.0)
    rewarded_force = torch.clamp(
        vertical_force - force_threshold,
        min=0.0,
        max=max_excess_force,
    )
    return torch.sum(rewarded_force, dim=1)


def feet_stumble_wmr(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return one when horizontal foot force exceeds vertical foot force."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_w = _torch(contact_sensor.data.net_forces_w)[:, sensor_cfg.body_ids]
    horizontal_force = torch.linalg.norm(forces_w[..., :2], dim=-1)
    vertical_force = torch.abs(forces_w[..., 2])
    return torch.any(horizontal_force > vertical_force, dim=1).float()


def flying_state_wmr(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_time_threshold: float,
) -> torch.Tensor:
    """Return one when neither foot has meaningful ground-contact time."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = _torch(contact_sensor.data.current_contact_time)[:, sensor_cfg.body_ids]
    return (torch.sum(contact_time, dim=1) < contact_time_threshold).float()


@configclass
class RewardsCfg:
    """WMR reward terms adapted to the G1 29-DoF joint and link names."""

    # Task rewards.
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # Base and control regularization.
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    energy = RewTerm(func=mdp.energy, weight=-0.001)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)

    # WMR joint-deviation groups from the paper appendix.
    joint_deviation_leg_pitch = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                    ".*_ankle_pitch_joint",
                ],
            )
        },
    )
    joint_deviation_hip_yaw_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                ],
            )
        },
    )
    joint_deviation_other = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_ankle_roll_joint",
                    "waist_.*_joint",
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            )
        },
    )

    # Foot rewards.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.2,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_force = RewTerm(
        func=feet_force_wmr,
        weight=5e-3,
        params={
            "force_threshold": 500.0,
            "max_excess_force": 400.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    feet_stumble = RewTerm(
        func=feet_stumble_wmr,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    flying_state = RewTerm(
        func=flying_state_wmr,
        weight=-1.0,
        params={
            "contact_time_threshold": 0.001,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class RobotEnvCfg(backpack_cfg.RobotEnvCfg):
    """Flat backpack training environment with WMR rewards."""

    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(backpack_cfg.RobotPlayEnvCfg):
    """Play environment matching :class:`RobotEnvCfg`."""

    rewards: RewardsCfg = RewardsCfg()
