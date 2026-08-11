"""Unitree-equivalent flat-backpack task with angular-velocity curriculum."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import velocity_env_cfg_flat_backpack_unitree as unitree_cfg


@configclass
class CurriculumCfg(unitree_cfg.CurriculumCfg):
    """Expand yaw-rate commands after the policy learns the current range."""

    ang_vel_cmd_levels = CurrTerm(func=mdp.ang_vel_cmd_levels)


@configclass
class RobotEnvCfg(unitree_cfg.RobotEnvCfg):
    """Training configuration with linear and angular velocity curricula."""

    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class RobotPlayEnvCfg(unitree_cfg.RobotPlayEnvCfg):
    """Play configuration matching the curve-training task."""

    curriculum: CurriculumCfg = CurriculumCfg()
