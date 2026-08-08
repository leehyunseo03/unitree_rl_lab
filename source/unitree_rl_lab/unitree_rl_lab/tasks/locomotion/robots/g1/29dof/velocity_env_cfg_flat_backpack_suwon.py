"""Flat G1 backpack task using the reward set documented in the Suwon handover PDF.

The PDF's ``unitree`` reward column matches ``velocity_env_cfg.RewardsCfg``.
This module combines that reward set with the existing flat-backpack scene,
payload randomization, commands, observations, actions, terminations, and
curriculum.
"""

from isaaclab.utils import configclass

from . import velocity_env_cfg as pdf_cfg
from . import velocity_env_cfg_flat_backpack as backpack_cfg


@configclass
class RewardsCfg(pdf_cfg.RewardsCfg):
    """Reward configuration from ``G1_Locomotion_인수인계.pdf``."""

    pass


@configclass
class RobotEnvCfg(backpack_cfg.RobotEnvCfg):
    """Flat backpack training environment with the Suwon PDF rewards."""

    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(backpack_cfg.RobotPlayEnvCfg):
    """Play environment matching :class:`RobotEnvCfg`."""

    rewards: RewardsCfg = RewardsCfg()
