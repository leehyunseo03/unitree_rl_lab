import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_flat:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_flat:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:FlatPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-Velocity-Flat-Backpack",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_flat_backpack:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_flat_backpack:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)


gym.register(
    id="Unitree-G1-29dof-Velocity-Flat-Backpack-Stand",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_flat_backpack_stand:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_flat_backpack_stand:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)


gym.register(
    id="Unitree-G1-29dof-Velocity-Flat-Backpack-Unitree",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_flat_backpack_unitree:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_flat_backpack_unitree:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


gym.register(
    id="Unitree-G1-29dof-Velocity-Flat-Backpack-V2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_flat_backpack_v2:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_flat_backpack_v2:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)


gym.register(
    id="Unitree-G1-29dof-Velocity-Flat-Backpack-V3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_flat_backpack_v3:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_flat_backpack_v3:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)


gym.register(
    id="Unitree-G1-29dof-Velocity-Flat-Backpack-Suwon",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_flat_backpack_suwon:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_flat_backpack_suwon:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)


gym.register(
    id="Unitree-G1-29dof-Velocity-Flat-Backpack-WMR",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_flat_backpack_wmr:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_flat_backpack_wmr:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)
