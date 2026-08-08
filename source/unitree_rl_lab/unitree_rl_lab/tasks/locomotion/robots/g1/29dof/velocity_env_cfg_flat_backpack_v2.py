"""Vertical-stability reward variant (v2) of the flat G1 backpack task.

The backpack environment inherits its rewards unchanged from
``velocity_env_cfg_flat``.  V2 keeps that complete reward set and changes only
the base-height and vertical-velocity penalty weights so the policy places
less importance on vertical position and velocity.

Deltas versus ``velocity_env_cfg_flat_backpack``::

    base_linear_velocity (Z velocity)   -2.0 -> -1.0
    base_height (target: 0.78 m)       -10.0 -> -5.0

All other rewards and all non-reward configuration remain identical to the
flat-backpack task.
"""

from isaaclab.utils import configclass

from . import velocity_env_cfg_flat as flat_cfg
from . import velocity_env_cfg_flat_backpack as backpack_cfg


_BACKPACK_REWARDS = flat_cfg.RewardsCfg()


@configclass
class RewardsCfg(flat_cfg.RewardsCfg):
    """Backpack rewards with only the two vertical penalties relaxed."""

    base_linear_velocity = _BACKPACK_REWARDS.base_linear_velocity.replace(weight=-1.0)
    base_height = _BACKPACK_REWARDS.base_height.replace(weight=-5.0)


@configclass
class RobotEnvCfg(backpack_cfg.RobotEnvCfg):
    """Flat backpack training configuration with v2 vertical rewards."""

    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(backpack_cfg.RobotPlayEnvCfg):
    """Play configuration matching :class:`RobotEnvCfg`."""

    rewards: RewardsCfg = RewardsCfg()
