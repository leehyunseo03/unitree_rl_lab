"""Reward-shaping variant (v2) of the flat G1 backpack velocity task.

Everything else -- scene, backpack geometry and payload randomization,
observations, actions, commands, terminations, curriculum -- is inherited
unchanged from :mod:`velocity_env_cfg_flat_backpack`.  Only reward weights
differ, plus one extra torso-rocking penalty defined below.

Intent of the shaping:

1. Emphasize velocity tracking so that *moving in the commanded direction*
   dominates the return, for every direction the command sampler produces
   (forward / backward / lateral / turning).
2. Reduce torso sway.  The backpack mass sits behind and above ``torso_link``,
   so torso rocking swings a large inertial moment around; that is the main
   source of instability the v1 weights do not price in directly.

Deltas versus v1 (everything else identical):

    track_lin_vel_xy        1.0    ->  1.5     +50%, tracking dominates
    track_ang_vel_z         0.5    ->  0.6
    base_angular_velocity  -0.05   -> -0.10    pelvis roll/pitch rate
    flat_orientation_l2    -5.0    -> -6.0     pelvis tilt
    joint_deviation_waists -1.0    -> -1.5     torso held over the pelvis
    torso_ang_vel_xy        --     -> -0.10    NEW: torso rocking rate
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import velocity_env_cfg_flat as flat_cfg
from . import velocity_env_cfg_flat_backpack as backpack_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def torso_ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
) -> torch.Tensor:
    """Penalize roll/pitch angular velocity of the backpack-carrying link.

    The stock ``ang_vel_xy_l2`` term only looks at the root (pelvis).  With the
    payload mounted behind and above ``torso_link``, the torso can rock about
    its own roll/pitch axes while the pelvis stays comparatively level -- that
    motion is unpriced by the v1 rewards and is exactly what shakes the
    backpack.  The world-frame angular velocity is rotated into the torso frame
    first so the penalty is orientation-invariant.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    ang_vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids]
    ang_vel_b = quat_apply_inverse(quat_w, ang_vel_w)
    return torch.sum(torch.square(ang_vel_b[..., :2]), dim=(-2, -1))


@configclass
class RewardsCfg(flat_cfg.RewardsCfg):
    """The v1 reward set with tracking emphasized and torso sway damped."""

    # New term.  Weight is deliberately in the same order as the pelvis
    # ang_vel penalty so it damps rocking without out-competing tracking.
    torso_ang_vel_xy = RewTerm(
        func=torso_ang_vel_xy_l2,
        weight=-0.10,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )

    def __post_init__(self):
        # -- velocity tracking gets a larger share of the return
        self.track_lin_vel_xy.weight = 1.5
        self.track_ang_vel_z.weight = 0.6

        # -- trunk stability
        self.base_angular_velocity.weight = -0.10
        self.flat_orientation_l2.weight = -6.0
        self.joint_deviation_waists.weight = -1.5


@configclass
class RobotEnvCfg(backpack_cfg.RobotEnvCfg):
    """Flat backpack training configuration with the v2 reward weights."""

    rewards: RewardsCfg = RewardsCfg()


@configclass
class RobotPlayEnvCfg(backpack_cfg.RobotPlayEnvCfg):
    """Play configuration matching :class:`RobotEnvCfg`."""

    rewards: RewardsCfg = RewardsCfg()
