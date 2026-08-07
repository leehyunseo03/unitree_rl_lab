# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher
from packaging.version import parse as parse_version

# local imports
import cli_args  # isort: skip
import wandb

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--export_only", action="store_true", help="Export the checkpoint and exit without playback.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def load_runner_checkpoint(runner, checkpoint_path: str, installed_rsl_rl_version: str):
    """Load both current and pre-v4 RSL-RL checkpoint formats."""
    if parse_version(installed_rsl_rl_version) < parse_version("4.0.0"):
        runner.load(checkpoint_path)
        return

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    legacy_state = checkpoint.get("model_state_dict")
    if legacy_state is None:
        runner.load(checkpoint_path)
        return

    actor_state = {}
    for key, value in legacy_state.items():
        if key == "std":
            actor_state["distribution.std_param"] = value
        elif key.startswith("actor."):
            actor_state[f"mlp.{key.removeprefix('actor.')}"] = value
        elif key.startswith("actor_obs_normalizer."):
            actor_state[f"obs_normalizer.{key.removeprefix('actor_obs_normalizer.')}"] = value

    if not actor_state:
        raise RuntimeError("Legacy checkpoint does not contain an actor state dictionary.")

    # Only the actor is needed for inference and export. The critic may have
    # been trained with a different set of privileged observations.
    runner.alg.actor.load_state_dict(actor_state, strict=True)
    runner.current_learning_iteration = checkpoint.get("iter", 0)
    print("[INFO]: Loaded legacy RSL-RL checkpoint into the current actor model.")


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    installed_rsl_rl_version = version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        # This helper is unavailable in older Isaac Lab releases. Import it only
        # when the corresponding option is requested so local checkpoints still work.
        from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    load_runner_checkpoint(runner, resume_path, installed_rsl_rl_version)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if parse_version(installed_rsl_rl_version) >= parse_version("4.0.0"):
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network module for legacy rsl-rl versions
        if parse_version(installed_rsl_rl_version) >= parse_version("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    print(f"[INFO]: Exported policy to: {os.path.join(export_model_dir, 'policy.onnx')}")
    if args_cli.export_only:
        env.close()
        return

    dt = env.unwrapped.step_dt
    
    
    #wandb init
    wandb.init(
        project="unitree_rl_lab_play",
        name=os.path.basename(log_dir)+"_play",
        config={
            "task": args_cli.task,
            "log_dir": log_dir,
            "checkpoint": resume_path,
        },
    )

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0
    
    
    # for torque env
    index = [3,9,4,10] # l_hip_roll, l_knee, r_hip_roll, r_knee
    robot = env.unwrapped.scene["robot"]
    
    max_torque = 0.0
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs) # policy output
            processed_actions = actions * 0.25 + 0.3 # aka target positions

            # terminal log, policy output actions
            # print(f"ACTIONS | l_hip_roll: {actions[0,3]:.2f} | l_knee: {actions[0,9]:.2f} | r_hip_roll: {actions[0,4]:.2f} | r_knee: {actions[0,10]:.2f}")
            
            # env stepping
            obs, _, _, _ = env.step(actions)
            
            current_max_tq = torch.max(torch.abs(robot.data.applied_torque)).item()
            if current_max_tq > max_torque:
                max_torque = current_max_tq
            torque = robot.data.applied_torque[0, index]
            # print(f"ALLTORQ | {robot.data.applied_torque[0]}") # log every joint's torque
            # print(f"TORQUES | l_hip_roll: {torque[0]:.2f} | l_knee: {torque[1]:.2f} | r_hip_roll: {torque[2]:.2f} | r_knee: {torque[3]:.2f} | Max Torque: {current_max_tq:.2f}")
            
            real_action = robot.data.joint_pos[0, index]
            joint_vel = robot.data.joint_vel[0, index]
            # proj_gravity = robot.data.projected_gravity_b[0]
            
            # wandb logging
            wandb.log({
                "action_l_hip_roll": actions[0,3].item(),
                "action_l_knee": actions[0,9].item(),
                "action_r_hip_roll": actions[0,4].item(),
                "action_r_knee": actions[0,10].item(),
                "real_action_l_hip_roll": real_action[0].item(),
                "real_action_l_knee": real_action[1].item(),
                "real_action_r_hip_roll": real_action[2].item(),
                "real_action_r_knee": real_action[3].item(),
                
                "processed_action_r_knee": processed_actions[0,10].item(),
                
                "joint_vel_l_knee": joint_vel[1].item(),
                "joint_vel_r_knee": joint_vel[3].item(),
                "torque_l_hip_roll": torque[0].item(),
                "torque_l_knee": torque[1].item(),
                "torque_r_hip_roll": torque[2].item(),
                "torque_r_knee": torque[3].item(),
            })
            
            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    wandb.finish()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
