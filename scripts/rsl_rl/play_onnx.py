# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Play an exported ONNX policy in a Unitree RL Lab environment."""

import argparse
import os
import pathlib
import sys
import time
from collections.abc import Mapping

from isaaclab.app import AppLauncher

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "source" / "unitree_rl_lab"
if str(SOURCE_PATH) not in sys.path:
    sys.path.insert(0, str(SOURCE_PATH))

parser = argparse.ArgumentParser(description="Play an exported ONNX policy in Isaac Lab.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--policy",
    type=str,
    default="deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx",
    help="Path to the exported ONNX policy, relative to the repo root or absolute.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time, if possible.")
parser.add_argument(
    "--command",
    type=float,
    nargs=3,
    metavar=("LIN_X", "LIN_Y", "ANG_Z"),
    default=None,
    help="Fixed base velocity command in the robot frame: x m/s, y m/s, yaw rad/s.",
)
parser.add_argument("--lin-vel-x", type=float, default=0.5, help="Fixed forward/backward velocity command in m/s.")
parser.add_argument("--lin-vel-y", type=float, default=0.0, help="Fixed left/right velocity command in m/s.")
parser.add_argument("--ang-vel-z", type=float, default=0.0, help="Fixed yaw velocity command in rad/s.")
parser.add_argument(
    "--no-camera-follow",
    action="store_true",
    default=False,
    help="Disable initial viewport camera placement on the robot.",
)
parser.add_argument(
    "--camera-follow",
    action="store_true",
    default=False,
    help="Continuously update the viewport camera to follow the robot.",
)
parser.add_argument(
    "--no-webrtc-dynamic-resize",
    action="store_true",
    default=False,
    help="Do not enable Kit's WebRTC dynamic stream resizing workaround.",
)
parser.add_argument(
    "--disable-base-contact-termination",
    action="store_true",
    default=False,
    help="Do not reset the env when the torso contacts the ground during ONNX playback.",
)
parser.add_argument("--no-reset-log", action="store_true", default=False, help="Do not print reset causes.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _livestream_enabled(args_cli):
    if args_cli.livestream >= 0:
        return args_cli.livestream > 0
    return int(os.environ.get("LIVESTREAM", 0)) > 0


def _append_kit_arg(args_cli, setting):
    kit_args = args_cli.kit_args or ""
    if setting not in kit_args:
        args_cli.kit_args = f"{kit_args} {setting}".strip()


LIVESTREAM_ENABLED = _livestream_enabled(args_cli)

if LIVESTREAM_ENABLED:
    args_cli.camera_follow = True
    if not getattr(args_cli, "visualizer_explicit", False):
        args_cli.visualizer = ["kit"]
    if not args_cli.no_webrtc_dynamic_resize:
        _append_kit_arg(
            args_cli,
            "--/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize=true",
        )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import onnxruntime as ort
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _as_policy_obs(observations):
    while isinstance(observations, tuple):
        observations = observations[0]
    while isinstance(observations, Mapping):
        observations = observations.get("policy", next(iter(observations.values())))
    return observations


def _update_camera(env, robot):
    root_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
    target = [float(root_pos[0]), float(root_pos[1]), float(root_pos[2] + 0.35)]
    eye = [target[0] + 3.0, target[1] + 3.0, target[2] + 1.3]
    env.unwrapped.sim.set_camera_view(eye=eye, target=target)


def _warm_up_livestream(env, frames=3):
    if not LIVESTREAM_ENABLED:
        return
    for _ in range(frames):
        env.unwrapped.sim.render()


def _resolve_command(args_cli):
    if args_cli.command is not None:
        return tuple(args_cli.command)
    return (args_cli.lin_vel_x, args_cli.lin_vel_y, args_cli.ang_vel_z)


def _set_fixed_base_velocity(env_cfg, command):
    if not hasattr(env_cfg, "commands") or not hasattr(env_cfg.commands, "base_velocity"):
        raise AttributeError("The selected task does not expose commands.base_velocity.")

    lin_vel_x, lin_vel_y, ang_vel_z = command
    base_velocity = env_cfg.commands.base_velocity
    base_velocity.ranges.lin_vel_x = (lin_vel_x, lin_vel_x)
    base_velocity.ranges.lin_vel_y = (lin_vel_y, lin_vel_y)
    base_velocity.ranges.ang_vel_z = (ang_vel_z, ang_vel_z)
    if hasattr(base_velocity, "limit_ranges"):
        base_velocity.limit_ranges.lin_vel_x = (lin_vel_x, lin_vel_x)
        base_velocity.limit_ranges.lin_vel_y = (lin_vel_y, lin_vel_y)
        base_velocity.limit_ranges.ang_vel_z = (ang_vel_z, ang_vel_z)
    base_velocity.heading_command = False
    base_velocity.rel_standing_envs = 0.0
    base_velocity.resampling_time_range = (1.0e9, 1.0e9)


def _disable_base_contact_termination(env_cfg):
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "base_contact"):
        env_cfg.terminations.base_contact = None


def _force_runtime_base_velocity(env, command):
    if not hasattr(env.unwrapped, "command_manager"):
        return
    try:
        command_term = env.unwrapped.command_manager.get_term("base_velocity")
    except KeyError:
        return

    command_tensor = torch.tensor(command, device=env.unwrapped.device, dtype=torch.float32)
    if hasattr(command_term, "vel_command_b"):
        command_term.vel_command_b[:] = command_tensor
    if hasattr(command_term, "is_standing_env"):
        command_term.is_standing_env[:] = False


def _log_reset_cause(env, dones, frame):
    if not dones.any() or not hasattr(env.unwrapped, "termination_manager"):
        return

    manager = env.unwrapped.termination_manager
    reset_count = int(dones.count_nonzero().item())
    causes = []
    for term_name in manager.active_terms:
        term_value = manager.get_term(term_name)
        count = int(term_value.count_nonzero().item())
        if count > 0:
            causes.append(f"{term_name}={count}")
    cause_text = ", ".join(causes) if causes else "unknown"
    print(f"[INFO] Reset at frame {frame}: envs={reset_count}, causes: {cause_text}")


def main():
    policy_path = args_cli.policy
    if not os.path.isabs(policy_path):
        policy_path = os.path.abspath(policy_path)
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"ONNX policy not found: {policy_path}")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    command = _resolve_command(args_cli)
    _set_fixed_base_velocity(env_cfg, command)
    if args_cli.disable_base_contact_termination:
        _disable_base_contact_termination(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)
    robot = env.unwrapped.scene["robot"]
    if not args_cli.no_camera_follow:
        _update_camera(env, robot)
    _warm_up_livestream(env)
    _force_runtime_base_velocity(env, command)

    session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[INFO] Loaded ONNX policy: {policy_path}")
    print(f"[INFO] ONNX input: {input_name} {session.get_inputs()[0].shape}")
    print(f"[INFO] ONNX output: {output_name} {session.get_outputs()[0].shape}")
    print(
        "[INFO] Fixed base velocity command: "
        f"x={command[0]:.3f} m/s, y={command[1]:.3f} m/s, yaw={command[2]:.3f} rad/s"
    )

    obs = _as_policy_obs(env.get_observations())
    dt = env.unwrapped.step_dt

    frame = 0
    while simulation_app.is_running():
        start_time = time.time()
        if args_cli.camera_follow and not args_cli.no_camera_follow and frame % 10 == 0:
            _update_camera(env, robot)
        frame += 1
        _force_runtime_base_velocity(env, command)
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        actions_np = session.run([output_name], {input_name: obs_np})[0]
        actions = torch.as_tensor(actions_np, device=env.unwrapped.device, dtype=torch.float32)
        obs, _, dones, _ = env.step(actions)
        if not args_cli.no_reset_log:
            _log_reset_cause(env, dones, frame)
        obs = _as_policy_obs(obs)

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
