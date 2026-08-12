"""Unitree flat-backpack task with lateral and curved-driving curricula.

The base Unitree task expands both forward/backward ``lin_vel_x`` and
left/right ``lin_vel_y`` command ranges through ``lin_vel_cmd_levels``.
This variant additionally expands ``ang_vel_z`` so the learned motion
progresses from straight/lateral travel to curved driving.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import velocity_env_cfg_flat_backpack_unitree as unitree_cfg


@configclass
class CurriculumCfg(unitree_cfg.CurriculumCfg):
    """Expand forward, lateral, and yaw-rate commands during training."""

    # Expands lin_vel_x and lin_vel_y together, including left/right travel.
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
    # Adds curved driving by expanding the yaw-rate command range.
    ang_vel_cmd_levels = CurrTerm(func=mdp.ang_vel_cmd_levels)


@configclass
class RobotEnvCfg(unitree_cfg.RobotEnvCfg):
    """Training configuration with lateral and curved-driving curricula."""

    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class RobotPlayEnvCfg(unitree_cfg.RobotPlayEnvCfg):
    """Play configuration matching the curve-training task."""

    curriculum: CurriculumCfg = CurriculumCfg()
