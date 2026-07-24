#!/usr/bin/env python3
# Copyright (c) 2026, Unitree RL Lab contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Run an exported ONNX policy and save per-joint torque metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import sys
import time
from collections.abc import Mapping
from datetime import datetime

from isaaclab.app import AppLauncher


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "source" / "unitree_rl_lab"
if str(SOURCE_PATH) not in sys.path:
    sys.path.insert(0, str(SOURCE_PATH))


parser = argparse.ArgumentParser(description="Collect torque metrics from an exported ONNX policy.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity-Backpack", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--policy",
    type=str,
    required=True,
    help="Path to the exported ONNX policy, relative to the repo root or absolute.",
)
parser.add_argument("--duration", type=float, default=20.0, help="Recorded duration in simulated seconds.")
parser.add_argument("--max-steps", type=int, default=None, help="Override duration with an exact number of sim steps.")
parser.add_argument("--warmup-steps", type=int, default=50, help="Steps to run before recording torque samples.")
parser.add_argument("--record-every", type=int, default=1, help="Record one sample every N sim steps.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time, if possible.")
parser.add_argument(
    "--command",
    type=float,
    nargs=3,
    metavar=("LIN_X", "LIN_Y", "ANG_Z"),
    default=None,
    help="Fixed base velocity command in the robot frame: x m/s, y m/s, yaw rad/s.",
)
parser.add_argument("--lin-vel-x", type=float, default=0.4, help="Fixed forward/backward velocity command in m/s.")
parser.add_argument("--lin-vel-y", type=float, default=0.0, help="Fixed left/right velocity command in m/s.")
parser.add_argument("--ang-vel-z", type=float, default=0.0, help="Fixed yaw velocity command in rad/s.")
parser.add_argument(
    "--output-dir",
    type=str,
    default=str(REPO_ROOT / "metrics" / "shared_runs"),
    help="Directory where timestamped metric folders are written.",
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

from g1_effort_limits import G1_29DOF_EFFORT_LIMIT_SOURCE, G1_29DOF_EFFORT_LIMIT_SOURCE_URL
from g1_effort_limits import g1_29dof_effort_limit_nm


def _as_policy_obs(observations):
    while isinstance(observations, tuple):
        observations = observations[0]
    while isinstance(observations, Mapping):
        observations = observations.get("policy", next(iter(observations.values())))
    return observations


def _resolve_path(path_value: str) -> pathlib.Path:
    path = pathlib.Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


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


def _tensor_to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "torch"):
        value = value.torch
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _get_joint_effort_limits(robot, joint_count: int) -> np.ndarray:
    for attr_name in ("joint_effort_limits", "effort_limits", "effort_limits_sim"):
        if hasattr(robot.data, attr_name):
            limits = _tensor_to_numpy(getattr(robot.data, attr_name))
            if limits is not None and limits.size:
                limits = np.squeeze(limits)
                if limits.ndim > 1:
                    limits = limits[0]
                if limits.shape[0] == joint_count:
                    return limits.astype(float)
    return np.full(joint_count, np.nan, dtype=float)


def _get_g1_effort_limits(joint_names: list[str]) -> np.ndarray:
    limits = []
    for joint_name in joint_names:
        limit = g1_29dof_effort_limit_nm(joint_name)
        limits.append(float(limit) if limit is not None else np.nan)
    return np.asarray(limits, dtype=float)


def _log_reset_cause(env, dones, frame, reset_counts):
    if not dones.any() or not hasattr(env.unwrapped, "termination_manager"):
        return

    manager = env.unwrapped.termination_manager
    for term_name in manager.active_terms:
        term_value = manager.get_term(term_name)
        count = int(term_value.count_nonzero().item())
        if count > 0:
            reset_counts[term_name] = reset_counts.get(term_name, 0) + count
    if not args_cli.no_reset_log:
        reset_count = int(dones.count_nonzero().item())
        causes = ", ".join(f"{name}={count}" for name, count in reset_counts.items()) or "unknown"
        print(f"[INFO] Reset at frame {frame}: envs={reset_count}, cumulative causes: {causes}")


def _slugify(value: str) -> str:
    safe_chars = []
    for char in value:
        if char.isalnum():
            safe_chars.append(char.lower())
        elif char in ("-", "_", "."):
            safe_chars.append(char.replace(".", "p"))
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("_")


def _make_run_dir(output_dir: pathlib.Path, task: str, policy_path: pathlib.Path, command) -> pathlib.Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_stem = _slugify(task)
    policy_stem = policy_path.parent.parent.name if policy_path.parent.name == "exported" else policy_path.stem
    command_tag = f"vx{command[0]:.2f}_vy{command[1]:.2f}_wz{command[2]:.2f}".replace("-", "m").replace(".", "p")
    run_dir = output_dir / f"{timestamp}_{task_stem}_{policy_stem}_{command_tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_timeseries(path: pathlib.Path, joint_names: list[str], samples: list[dict]):
    fieldnames = ["frame", "time_s", *joint_names]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            row = {"frame": sample["frame"], "time_s": f"{sample['time_s']:.6f}"}
            row.update({name: f"{value:.6f}" for name, value in zip(joint_names, sample["torques"])})
            writer.writerow(row)


def _write_summary(path: pathlib.Path, summary_rows: list[dict]):
    fieldnames = [
        "joint_index",
        "joint_name",
        "min_torque_nm",
        "max_torque_nm",
        "mean_torque_nm",
        "mean_abs_torque_nm",
        "rms_torque_nm",
        "peak_abs_torque_nm",
        "effort_limit_nm",
        "peak_abs_over_limit_pct",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def _write_markdown_table(path: pathlib.Path, summary_rows: list[dict]):
    columns = [
        "joint_index",
        "joint_name",
        "min_torque_nm",
        "max_torque_nm",
        "mean_abs_torque_nm",
        "rms_torque_nm",
        "peak_abs_torque_nm",
        "effort_limit_nm",
        "peak_abs_over_limit_pct",
    ]
    with path.open("w") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in summary_rows:
            f.write("| " + " | ".join(str(row[column]) for column in columns) + " |\n")


def _plot_torque_ranges(path: pathlib.Path, joint_names: list[str], torque_min, torque_max, effort_limits) -> str:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        svg_path = path.with_suffix(".svg")
        _write_svg_torque_ranges(svg_path, joint_names, torque_min, torque_max, effort_limits)
        print(f"[WARN] matplotlib unavailable ({exc}); wrote SVG graph instead: {svg_path}")
        return str(svg_path)

    y = np.arange(len(joint_names))
    fig_height = max(8.0, len(joint_names) * 0.32)
    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    ax.hlines(y, torque_min, torque_max, color="#2563eb", linewidth=4, label="observed min to max")
    ax.scatter(torque_min, y, color="#1d4ed8", s=16)
    ax.scatter(torque_max, y, color="#1d4ed8", s=16)

    finite_limits = np.isfinite(effort_limits)
    if finite_limits.any():
        ax.hlines(
            y[finite_limits],
            -effort_limits[finite_limits],
            effort_limits[finite_limits],
            color="#d1d5db",
            linewidth=1.5,
            label="official G1 effort limit",
            zorder=0,
        )

    ax.axvline(0.0, color="#111827", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(joint_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Torque (N*m)")
    ax.set_title("Observed Joint Torque Range")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="lower right")
    x_min = float(np.nanmin(torque_min))
    x_max = float(np.nanmax(torque_max))
    if finite_limits.any():
        x_min = min(x_min, float(np.nanmin(-effort_limits[finite_limits])))
        x_max = max(x_max, float(np.nanmax(effort_limits[finite_limits])))
    pad = max((x_max - x_min) * 0.05, 1.0)
    ax.set_xlim(x_min - pad, x_max + pad)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _write_svg_torque_ranges(path: pathlib.Path, joint_names: list[str], torque_min, torque_max, effort_limits):
    width = 1100
    row_h = 24
    left = 280
    right = 30
    top = 35
    height = top + row_h * len(joint_names) + 40
    finite_limits = np.isfinite(effort_limits)
    x_min = float(np.nanmin(torque_min))
    x_max = float(np.nanmax(torque_max))
    if finite_limits.any():
        x_min = min(x_min, float(np.nanmin(-effort_limits[finite_limits])))
        x_max = max(x_max, float(np.nanmax(effort_limits[finite_limits])))
    pad = max((x_max - x_min) * 0.05, 1.0)
    x_min -= pad
    x_max += pad

    def scale_x(value):
        return left + (float(value) - x_min) / (x_max - x_min) * (width - left - right)

    zero_x = scale_x(0.0)
    with path.open("w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n')
        f.write('<rect width="100%" height="100%" fill="white"/>\n')
        f.write(f'<line x1="{zero_x:.1f}" y1="20" x2="{zero_x:.1f}" y2="{height - 20}" stroke="#111827"/>\n')
        for idx, name in enumerate(joint_names):
            y = top + idx * row_h
            x1 = scale_x(torque_min[idx])
            x2 = scale_x(torque_max[idx])
            f.write(f'<text x="8" y="{y + 4}" font-size="12" font-family="monospace">{name}</text>\n')
            limit = effort_limits[idx]
            if np.isfinite(limit):
                f.write(
                    f'<line x1="{scale_x(-limit):.1f}" y1="{y}" x2="{scale_x(limit):.1f}" '
                    f'y2="{y}" stroke="#d1d5db" stroke-width="1.5"/>\n'
                )
            f.write(f'<line x1="{x1:.1f}" y1="{y}" x2="{x2:.1f}" y2="{y}" stroke="#2563eb" stroke-width="4"/>\n')
            f.write(f'<circle cx="{x1:.1f}" cy="{y}" r="3" fill="#1d4ed8"/>\n')
            f.write(f'<circle cx="{x2:.1f}" cy="{y}" r="3" fill="#1d4ed8"/>\n')
        f.write("</svg>\n")


def _format_float(value, digits=4):
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.{digits}f}"


def main():
    policy_path = _resolve_path(args_cli.policy)
    if not policy_path.exists():
        raise FileNotFoundError(f"ONNX policy not found: {policy_path}")

    command = _resolve_command(args_cli)
    output_dir = _resolve_path(args_cli.output_dir)
    run_dir = _make_run_dir(output_dir, args_cli.task, policy_path, command)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    _set_fixed_base_velocity(env_cfg, command)
    if args_cli.disable_base_contact_termination:
        _disable_base_contact_termination(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)
    robot = env.unwrapped.scene["robot"]
    _force_runtime_base_velocity(env, command)

    session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    dt = float(env.unwrapped.step_dt)
    max_steps = args_cli.max_steps if args_cli.max_steps is not None else int(math.ceil(args_cli.duration / dt))
    max_steps = max(max_steps, 1)
    record_every = max(args_cli.record_every, 1)

    print(f"[INFO] Loaded ONNX policy: {policy_path}")
    print(f"[INFO] Task: {args_cli.task}")
    print(
        "[INFO] Fixed base velocity command: "
        f"x={command[0]:.3f} m/s, y={command[1]:.3f} m/s, yaw={command[2]:.3f} rad/s"
    )
    print(f"[INFO] Recording {max_steps} steps after {args_cli.warmup_steps} warmup steps.")
    print(f"[INFO] Saving torque report to: {run_dir}")

    obs = _as_policy_obs(env.get_observations())
    samples = []
    reset_counts = {}
    total_steps = args_cli.warmup_steps + max_steps
    frame = 0

    while simulation_app.is_running() and frame < total_steps:
        start_time = time.time()
        _force_runtime_base_velocity(env, command)
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        actions_np = session.run([output_name], {input_name: obs_np})[0]
        actions = torch.as_tensor(actions_np, device=env.unwrapped.device, dtype=torch.float32)
        obs, _, dones, _ = env.step(actions)
        frame += 1
        _log_reset_cause(env, dones, frame, reset_counts)
        obs = _as_policy_obs(obs)

        if frame > args_cli.warmup_steps and (frame - args_cli.warmup_steps - 1) % record_every == 0:
            torque = robot.data.applied_torque[0].detach().cpu().numpy().astype(float)
            samples.append({"frame": frame, "time_s": (frame - args_cli.warmup_steps) * dt, "torques": torque})

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    if not samples:
        env.close()
        raise RuntimeError("No torque samples were collected. Increase --duration or lower --warmup-steps.")

    joint_names = list(robot.data.joint_names)
    torque_matrix = np.stack([sample["torques"] for sample in samples], axis=0)
    sim_effort_limits = _get_joint_effort_limits(robot, len(joint_names))
    g1_effort_limits = _get_g1_effort_limits(joint_names)
    effort_limits = np.where(np.isfinite(g1_effort_limits), g1_effort_limits, sim_effort_limits)
    torque_min = torque_matrix.min(axis=0)
    torque_max = torque_matrix.max(axis=0)
    torque_mean = torque_matrix.mean(axis=0)
    torque_mean_abs = np.abs(torque_matrix).mean(axis=0)
    torque_rms = np.sqrt(np.square(torque_matrix).mean(axis=0))
    torque_peak_abs = np.abs(torque_matrix).max(axis=0)

    summary_rows = []
    for idx, joint_name in enumerate(joint_names):
        limit = effort_limits[idx]
        usage_pct = torque_peak_abs[idx] / limit * 100.0 if np.isfinite(limit) and limit > 0.0 else np.nan
        summary_rows.append(
            {
                "joint_index": idx,
                "joint_name": joint_name,
                "min_torque_nm": _format_float(torque_min[idx]),
                "max_torque_nm": _format_float(torque_max[idx]),
                "mean_torque_nm": _format_float(torque_mean[idx]),
                "mean_abs_torque_nm": _format_float(torque_mean_abs[idx]),
                "rms_torque_nm": _format_float(torque_rms[idx]),
                "peak_abs_torque_nm": _format_float(torque_peak_abs[idx]),
                "effort_limit_nm": _format_float(limit),
                "peak_abs_over_limit_pct": _format_float(usage_pct, digits=2),
            }
        )

    _write_timeseries(run_dir / "torque_timeseries.csv", joint_names, samples)
    _write_summary(run_dir / "torque_summary.csv", summary_rows)
    _write_markdown_table(run_dir / "torque_summary.md", summary_rows)
    graph_path = _plot_torque_ranges(run_dir / "torque_range.png", joint_names, torque_min, torque_max, effort_limits)

    metadata = {
        "task": args_cli.task,
        "policy": str(policy_path),
        "command": {"lin_vel_x": command[0], "lin_vel_y": command[1], "ang_vel_z": command[2]},
        "step_dt": dt,
        "warmup_steps": args_cli.warmup_steps,
        "recorded_steps": len(samples),
        "record_every": record_every,
        "reset_counts": reset_counts,
        "onnx_input": {"name": input_name, "shape": session.get_inputs()[0].shape},
        "onnx_output": {"name": output_name, "shape": session.get_outputs()[0].shape},
        "effort_limit_source": G1_29DOF_EFFORT_LIMIT_SOURCE,
        "effort_limit_source_url": G1_29DOF_EFFORT_LIMIT_SOURCE_URL,
        "outputs": {
            "summary_csv": str(run_dir / "torque_summary.csv"),
            "summary_markdown": str(run_dir / "torque_summary.md"),
            "timeseries_csv": str(run_dir / "torque_timeseries.csv"),
            "range_graph": graph_path,
        },
    }
    with (run_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Saved summary table: {run_dir / 'torque_summary.csv'}")
    print(f"[INFO] Saved markdown table: {run_dir / 'torque_summary.md'}")
    print(f"[INFO] Saved time series: {run_dir / 'torque_timeseries.csv'}")
    print(f"[INFO] Saved torque range graph: {graph_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
