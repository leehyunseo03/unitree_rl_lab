import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import velocity_env_cfg as base_cfg


BACKPACK_SIZE = (0.06, 0.14, 0.12)
BACKPACK_BACK_SURFACE_X = -0.085
BACKPACK_LOCAL_POS = (BACKPACK_BACK_SURFACE_X - 0.5 * BACKPACK_SIZE[0], 0.0, 0.15)
BACKPACK_MASS_RANGE = (0.6, 1.0)
BACKPACK_PLAY_MASS = (0.8, 0.8)
BACKPACK_COM_X_RANDOMIZATION = 0.01
BACKPACK_COM_Y_RANDOMIZATION = 0.02
BACKPACK_COM_Z_RANDOMIZATION = 0.01
BACKPACK_COM_RANGE = {
    "x": (
        BACKPACK_LOCAL_POS[0] - BACKPACK_COM_X_RANDOMIZATION,
        BACKPACK_LOCAL_POS[0] + BACKPACK_COM_X_RANDOMIZATION,
    ),
    "y": (-BACKPACK_COM_Y_RANDOMIZATION, BACKPACK_COM_Y_RANDOMIZATION),
    "z": (
        BACKPACK_LOCAL_POS[2] - BACKPACK_COM_Z_RANDOMIZATION,
        BACKPACK_LOCAL_POS[2] + BACKPACK_COM_Z_RANDOMIZATION,
    ),
}
BACKPACK_PLAY_COM = {
    "x": (BACKPACK_LOCAL_POS[0], BACKPACK_LOCAL_POS[0]),
    "y": (0.0, 0.0),
    "z": (BACKPACK_LOCAL_POS[2], BACKPACK_LOCAL_POS[2]),
}


@configclass
class RobotSceneCfg(base_cfg.RobotSceneCfg):
    """Flat G1 scene with a simple backpack box attached to the torso link."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            "TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    backpack = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/backpack_box",
        init_state=AssetBaseCfg.InitialStateCfg(pos=BACKPACK_LOCAL_POS),
        spawn=sim_utils.CuboidCfg(
            size=BACKPACK_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.14, 0.18), roughness=0.85),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )


@configclass
class EventCfg(base_cfg.EventCfg):
    """Payload events for the backpack task."""

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": BACKPACK_MASS_RANGE,
            "operation": "add",
        },
    )
    backpack_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": BACKPACK_COM_RANGE,
        },
    )


@configclass
class CommandsCfg(base_cfg.CommandsCfg):
    """Forward-only velocity commands for flat backpack walking."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.1),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.6),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        vel_xy_success_threshold=0.35,
        vel_yaw_success_threshold=0.2,
    )


_BASE_REWARDS = base_cfg.RewardsCfg()


@configclass
class RewardsCfg(base_cfg.RewardsCfg):
    """Backpack reward weights tuned toward stable forward walking."""

    track_lin_vel_xy = _BASE_REWARDS.track_lin_vel_xy.replace(weight=2.0)
    track_ang_vel_z = _BASE_REWARDS.track_ang_vel_z.replace(weight=1.5)
    base_angular_velocity = _BASE_REWARDS.base_angular_velocity.replace(weight=-0.10)
    action_rate = _BASE_REWARDS.action_rate.replace(weight=-0.03)
    flat_orientation_l2 = _BASE_REWARDS.flat_orientation_l2.replace(weight=-3.0)
    gait = _BASE_REWARDS.gait.replace(weight=0.3)
    feet_air_time = _BASE_REWARDS.feet_air_time.replace(weight=0.05)


@configclass
class CurriculumCfg(base_cfg.CurriculumCfg):
    """No terrain curriculum for the flat backpack task."""

    terrain_levels = None
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(base_cfg.RobotEnvCfg):
    """Flat locomotion environment with an 800 g backpack payload model."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.events.add_base_mass.params["mass_distribution_params"] = BACKPACK_PLAY_MASS
        self.events.backpack_com.params["com_range"] = BACKPACK_PLAY_COM
        self.events.base_external_force_torque = None
        self.events.push_robot = None
