import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import velocity_env_cfg as base_cfg


BACKPACK_BACK_SURFACE_X = -0.070
BACKPACK_CENTER_Y = 0.0
BACKPACK_CENTER_Z = 0.13

BACKPACK_PLATE_THICKNESS = 0.003
BACKPACK_PLATE_WIDTH = 0.135
BACKPACK_PLATE_HEIGHT = 0.24
BACKPACK_COMPONENT_GAP = 0.01

BACKPACK_PLATE_SIZE = (BACKPACK_PLATE_THICKNESS, BACKPACK_PLATE_WIDTH, BACKPACK_PLATE_HEIGHT)
BACKPACK_CONVERTER_SIZE = (0.05, 0.11, 0.05)
BACKPACK_JETSON_ORIN_SIZE = (0.05, 0.11, 0.11)
BACKPACK_BATTERY_SIZE = (0.05, 0.11, 0.06)

BACKPACK_PLATE_MASS = 0.150
BACKPACK_CONVERTER_MASS = 0.100
BACKPACK_JETSON_ORIN_MASS = 0.611
BACKPACK_BATTERY_MASS = 0.440

BACKPACK_CONVERTER_Z_OFFSET = 0.5 * BACKPACK_PLATE_HEIGHT - 0.5 * BACKPACK_CONVERTER_SIZE[2]
BACKPACK_JETSON_ORIN_Z_OFFSET = (
    BACKPACK_CONVERTER_Z_OFFSET
    - 0.5 * BACKPACK_CONVERTER_SIZE[2]
    - BACKPACK_COMPONENT_GAP
    - 0.5 * BACKPACK_JETSON_ORIN_SIZE[2]
)
BACKPACK_BATTERY_Z_OFFSET = (
    BACKPACK_JETSON_ORIN_Z_OFFSET
    - 0.5 * BACKPACK_JETSON_ORIN_SIZE[2]
    - BACKPACK_COMPONENT_GAP
    - 0.5 * BACKPACK_BATTERY_SIZE[2]
)


def _plate_local_pos():
    return (
        BACKPACK_BACK_SURFACE_X - 0.5 * BACKPACK_PLATE_THICKNESS,
        BACKPACK_CENTER_Y,
        BACKPACK_CENTER_Z,
    )


def _component_local_pos(size, z_offset):
    return (
        BACKPACK_BACK_SURFACE_X - BACKPACK_PLATE_THICKNESS - 0.5 * size[0],
        BACKPACK_CENTER_Y,
        BACKPACK_CENTER_Z + z_offset,
    )


BACKPACK_PLATE_POS = _plate_local_pos()
BACKPACK_CONVERTER_POS = _component_local_pos(BACKPACK_CONVERTER_SIZE, BACKPACK_CONVERTER_Z_OFFSET)
BACKPACK_JETSON_ORIN_POS = _component_local_pos(BACKPACK_JETSON_ORIN_SIZE, BACKPACK_JETSON_ORIN_Z_OFFSET)
BACKPACK_BATTERY_POS = _component_local_pos(BACKPACK_BATTERY_SIZE, BACKPACK_BATTERY_Z_OFFSET)

BACKPACK_PARTS = (
    {"name": "plate", "pos": BACKPACK_PLATE_POS, "size": BACKPACK_PLATE_SIZE, "mass": BACKPACK_PLATE_MASS},
    {
        "name": "converter",
        "pos": BACKPACK_CONVERTER_POS,
        "size": BACKPACK_CONVERTER_SIZE,
        "mass": BACKPACK_CONVERTER_MASS,
    },
    {
        "name": "jetson_orin",
        "pos": BACKPACK_JETSON_ORIN_POS,
        "size": BACKPACK_JETSON_ORIN_SIZE,
        "mass": BACKPACK_JETSON_ORIN_MASS,
    },
    {"name": "battery", "pos": BACKPACK_BATTERY_POS, "size": BACKPACK_BATTERY_SIZE, "mass": BACKPACK_BATTERY_MASS},
)
BACKPACK_TOTAL_MASS = sum(part["mass"] for part in BACKPACK_PARTS)
BACKPACK_LOCAL_COM = tuple(
    sum(part["pos"][axis] * part["mass"] for part in BACKPACK_PARTS) / BACKPACK_TOTAL_MASS for axis in range(3)
)
BACKPACK_MASS_RANDOMIZATION = 0.2
BACKPACK_TRAIN_MASS_RANGE = (
    BACKPACK_TOTAL_MASS - BACKPACK_MASS_RANDOMIZATION,
    BACKPACK_TOTAL_MASS + BACKPACK_MASS_RANDOMIZATION,
)
BACKPACK_PLAY_MASS_RANGE = (BACKPACK_TOTAL_MASS, BACKPACK_TOTAL_MASS)
BACKPACK_TRAIN_POS_OFFSET_RANGE = {
    "x": (0.0, 0.0),
    "y": (-0.04, 0.04),
    "z": (-0.05, 0.05),
}
BACKPACK_PLAY_POS_OFFSET_RANGE = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
}

GROUND_STATIC_FRICTION_RANGE = (0.45, 1.35)
GROUND_DYNAMIC_FRICTION_RANGE = (0.35, 1.10)
GROUND_RESTITUTION_RANGE = (0.0, 0.02)
GROUND_MATERIAL_BUCKETS = 96

SOFT_GROUND_CONTACT_STIFFNESS = 2.5e5
SOFT_GROUND_CONTACT_DAMPING = 5.0e3


@configclass
class RobotSceneCfg(base_cfg.RobotSceneCfg):
    """Flat G1 scene with a fixed backpack payload and compliant ground contact."""

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
            restitution=0.0,
            compliant_contact_stiffness=SOFT_GROUND_CONTACT_STIFFNESS,
            compliant_contact_damping=SOFT_GROUND_CONTACT_DAMPING,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            "TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    backpack_plate = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/backpack_plate",
        init_state=AssetBaseCfg.InitialStateCfg(pos=BACKPACK_PLATE_POS),
        spawn=sim_utils.CuboidCfg(
            size=BACKPACK_PLATE_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02, 0.025, 0.03), roughness=0.9),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )
    backpack_converter = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/backpack_converter",
        init_state=AssetBaseCfg.InitialStateCfg(pos=BACKPACK_CONVERTER_POS),
        spawn=sim_utils.CuboidCfg(
            size=BACKPACK_CONVERTER_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.20, 0.22), roughness=0.75),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )
    backpack_jetson_orin = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/backpack_jetson_orin",
        init_state=AssetBaseCfg.InitialStateCfg(pos=BACKPACK_JETSON_ORIN_POS),
        spawn=sim_utils.CuboidCfg(
            size=BACKPACK_JETSON_ORIN_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.18, 0.12), roughness=0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )
    backpack_battery = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/backpack_battery",
        init_state=AssetBaseCfg.InitialStateCfg(pos=BACKPACK_BATTERY_POS),
        spawn=sim_utils.CuboidCfg(
            size=BACKPACK_BATTERY_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.055, 0.06), roughness=0.85),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )


@configclass
class EventCfg(base_cfg.EventCfg):
    """Payload and ground-contact randomization for the backpack task."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": GROUND_STATIC_FRICTION_RANGE,
            "dynamic_friction_range": GROUND_DYNAMIC_FRICTION_RANGE,
            "restitution_range": GROUND_RESTITUTION_RANGE,
            "num_buckets": GROUND_MATERIAL_BUCKETS,
            "make_consistent": True,
        },
    )

    add_base_mass = None

    fixed_backpack_payload = EventTerm(
        func=mdp.apply_fixed_payload_to_rigid_body,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "payload_parts": BACKPACK_PARTS,
            "payload_mass_range": BACKPACK_TRAIN_MASS_RANGE,
            "payload_pos_offset_range": BACKPACK_TRAIN_POS_OFFSET_RANGE,
        },
    )


@configclass
class CommandsCfg(base_cfg.CommandsCfg):
    """Omnidirectional velocity commands for flat backpack walking."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.6),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.5, 0.5),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.6),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.5, 0.5),
        ),
        vel_xy_success_threshold=0.35,
        vel_yaw_success_threshold=0.4,
    )


_BASE_REWARDS = base_cfg.RewardsCfg()


@configclass
class RewardsCfg(base_cfg.RewardsCfg):
    """Backpack reward weights tuned toward stable omnidirectional walking."""

    track_lin_vel_xy = _BASE_REWARDS.track_lin_vel_xy.replace(weight=2.0)
    track_ang_vel_z = _BASE_REWARDS.track_ang_vel_z.replace(weight=1.0)
    alive = RewTerm(func=mdp.is_alive, weight=0.25)
    termination_penalty = _BASE_REWARDS.termination_penalty.replace(weight=-250.0)
    base_linear_velocity = _BASE_REWARDS.base_linear_velocity.replace(weight=-1.0)
    base_angular_velocity = _BASE_REWARDS.base_angular_velocity.replace(weight=-0.07)
    action_rate = _BASE_REWARDS.action_rate.replace(weight=-0.03)
    dof_pos_limits = _BASE_REWARDS.dof_pos_limits.replace(weight=-3.0)
    flat_orientation_l2 = _BASE_REWARDS.flat_orientation_l2.replace(weight=-3.0)
    base_height = _BASE_REWARDS.base_height.replace(weight=-3.0)
    gait = _BASE_REWARDS.gait.replace(weight=0.6)
    feet_slide = _BASE_REWARDS.feet_slide.replace(weight=-0.3)
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.4,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.08,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )
    feet_air_time = _BASE_REWARDS.feet_air_time.replace(weight=0.10)


@configclass
class TerminationsCfg(base_cfg.TerminationsCfg):
    """Terminate severely tilted backpack rollouts early."""

    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg(base_cfg.CurriculumCfg):
    """No terrain curriculum for the flat backpack task."""

    terrain_levels = None
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(base_cfg.RobotEnvCfg):
    """Flat locomotion environment with backpack payload and ground-contact DR."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.events.fixed_backpack_payload.params["payload_mass_range"] = BACKPACK_PLAY_MASS_RANGE
        self.events.fixed_backpack_payload.params["payload_pos_offset_range"] = BACKPACK_PLAY_POS_OFFSET_RANGE
        self.events.base_external_force_torque = None
        self.events.push_robot = None
