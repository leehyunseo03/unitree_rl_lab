# ONNX Policy Torque Metrics

`onnx_torque_report.py` runs an exported ONNX policy with the same fixed forward-walking command style as `scripts/rsl_rl/play_onnx.py`, records every joint's applied torque, and saves a table plus a torque-range graph.

## Output Location

Each run creates a timestamped folder under:

```bash
/home/hslee/IsaacLab_ws/unitree_rl_lab/metrics/shared_runs/
```

If you run inside the container from `/workspace/unitree_rl_lab`, the same files appear under:

```bash
/workspace/unitree_rl_lab/metrics/shared_runs/
```

This directory is intentionally writable by the container user. On the host it should look like:

```bash
drwxrwxrwx ... /home/hslee/IsaacLab_ws/unitree_rl_lab/metrics/shared_runs
```

The saved files are:

- `torque_summary.csv`: one row per joint with min/max/mean/RMS/peak torque and official Unitree G1 effort-limit usage.
- `torque_summary.md`: the same summary as a Markdown table for quick reading.
- `torque_timeseries.csv`: per-step torque values for every joint.
- `torque_timeseries_grid.png`: a 3-by-10 grid with one time-series graph per joint and the final cell left empty. Each graph uses simulation time on the x-axis and applied torque on the y-axis. If `matplotlib` is unavailable, `torque_timeseries_grid.svg` is written instead.
- `joint_position_timeseries.csv` and `joint_position_timeseries_grid.png`: joint position in radians for all 29 joints.
- `joint_velocity_timeseries.csv` and `joint_velocity_timeseries_grid.png`: joint velocity in radians per second for all 29 joints.
- `base_angular_velocity_timeseries.csv` and `base_angular_velocity_timeseries_grid.png`: body-frame roll, pitch, and yaw angular velocity in a 1-by-3 grid.
- `policy_action_timeseries.csv` and `policy_action_timeseries_grid.png`: raw ONNX policy actions for all 29 joints.
- `torque_range.png`: graph showing each joint's observed min torque to max torque range. The gray bars are official Unitree G1 29DOF URDF effort limits. If `matplotlib` is unavailable, `torque_range.svg` is written instead.
- `metadata.json`: task, policy path, command, sim dt, reset counts, and output paths.

## Run

From the repository root in the Isaac Lab container:

```bash
/isaac-sim/python.sh metrics/onnx_torque_report.py \
  --task Unitree-G1-29dof-Velocity-Backpack \
  --policy /workspace/unitree_rl_lab/container_runs/rsl_rl/unitree_g1_29dof_velocity_backpack/2026-07-13_13-26-45_g1_29dof_velocity_backpack_2kg_dr_pm02_reward_v2_yaw04/exported/policy.onnx \
  --lin-vel-x 0.4 \
  --duration 20 \
  --headless
```

To match the existing playback command more closely while watching it:

```bash
LIVESTREAM=2 /isaac-sim/python.sh metrics/onnx_torque_report.py \
  --task Unitree-G1-29dof-Velocity-Backpack \
  --policy /workspace/unitree_rl_lab/container_runs/rsl_rl/unitree_g1_29dof_velocity_backpack/2026-07-25_18-25-54_g1_29dof_velocity_backpack_1p30kg_dr_pm02_y4cm_z5cm_omni_floor_dr_stable_resume_to_100k/exported/policy.onnx \
  --command 0.4 0.0 0.0 \
  --seed 42 \
  --duration 20 \
  --real-time \
  --livestream 2
```

Use `--duration` to control the recorded simulated seconds. Startup transients are skipped by `--warmup-steps 50` by default.

By default, the metric collector applies the same deterministic evaluation settings to Plain and Backpack runs:

- flat rigid ground with fixed friction;
- observation corruption disabled;
- startup physics and base-mass randomization disabled;
- deterministic initial pose and joint velocity;
- the same seed;
- episode timeout disabled so reset transients are not recorded.

The intended differences are the ONNX policy and the fixed backpack payload. Use
`--preserve-task-eval-settings` only when the original task-specific play configuration is required instead.

## Run Without Backpack

Use the same script with the non-backpack task:

```bash
/isaac-sim/python.sh metrics/onnx_torque_report.py \
  --task Unitree-G1-29dof-Velocity \
  --policy /workspace/unitree_rl_lab/container_runs/rsl_rl/unitree_g1_29dof_velocity_backpack/2026-07-13_13-26-45_g1_29dof_velocity_backpack_2kg_dr_pm02_reward_v2_yaw04/exported/policy.onnx \
  --lin-vel-x 0.4 \
  --duration 20 \
  --headless
```

That command evaluates the same ONNX policy in the plain G1 environment, without the backpack mass and COM events. If you want to evaluate a separately trained plain-G1 policy instead, keep `--task Unitree-G1-29dof-Velocity` and replace `--policy` with that policy's `exported/policy.onnx`.

## Compare Two Runs

After creating one backpack run and one no-backpack run, compare their torque ranges:

```bash
python3 metrics/compare_torque_ranges.py
```

With no arguments, the script automatically compares the latest two folders under `metrics/shared_runs/` that contain `torque_summary.csv`. The output is saved under:

```bash
/workspace/unitree_rl_lab/metrics/shared_runs/comparisons/
```

To compare specific runs:

```bash
python3 metrics/compare_torque_ranges.py \
  --run-a no_backpack \
  --run-b 20260727_041253_unitree-g1-29dof-velocity-backpack_2026-07-25_18-25-54_g1_29dof_velocity_backpack_1p30kg_dr_pm02_y4cm_z5cm_omni_floor_dr_stable_resume_to_100k_vx0p40_vy0p00_wz0p00 \
  --label-a Plain-G1 \
  --label-b Backpack
```
The comparison output includes:

- `*_torque_range_compare.png`: blue and red min-to-max torque ranges on the same joint axis.
- `*_<run-label>_<signal>_timeseries_grid.png`: separate Plain and Backpack grids for torque, joint position, joint velocity, base angular velocity, and policy action. Matching components use the same y-axis range in both files.
- `*_torque_range_compare_delta.csv`: peak absolute torque difference per joint.

## Export ONNX First

If `exported/policy.onnx` does not exist yet:

```bash
/isaac-sim/python.sh scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Velocity-Backpack \
  --checkpoint /workspace/unitree_rl_lab/container_runs/rsl_rl/unitree_g1_29dof_velocity_backpack/2026-07-13_13-26-45_g1_29dof_velocity_backpack_2kg_dr_pm02_reward_v2_yaw04/model_49999.pt \
  --num_envs 1 \
  --headless \
  --export-only
```

## Record and Compare Plain vs Backpack

Run all three steps in the same shell so both recordings use the same `RUN_ROOT`.
The `/workspace/unitree_rl_lab` container path is mounted at
`/home/hslee/IsaacLab_ws/unitree_rl_lab` on the host.

```bash
cd /workspace/unitree_rl_lab

RUN_ROOT=/workspace/unitree_rl_lab/metrics/shared_runs/plain_original_vs_backpack_1p30kg_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT"

# 1. Backpack policy + Backpack env
#    This also writes torque_timeseries_grid.png in the new run directory.
/isaac-sim/python.sh metrics/onnx_torque_report.py \
  --task Unitree-G1-29dof-Velocity-Backpack \
  --policy /workspace/unitree_rl_lab/container_runs/rsl_rl/unitree_g1_29dof_velocity_backpack/2026-07-25_18-25-54_g1_29dof_velocity_backpack_1p30kg_dr_pm02_y4cm_z5cm_omni_floor_dr_stable_resume_to_100k/exported/policy.onnx \
  --command 0.4 0.0 0.0 \
  --seed 42 \
  --duration 20 \
  --output-dir "$RUN_ROOT" \
  --headless

# 2. 원래 Plain policy + Plain env
#    This also writes torque_timeseries_grid.png in the new run directory.
/isaac-sim/python.sh metrics/onnx_torque_report.py \
  --task Unitree-G1-29dof-Velocity \
  --policy /workspace/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx \
  --command 0.4 0.0 0.0 \
  --seed 42 \
  --duration 20 \
  --output-dir "$RUN_ROOT" \
  --headless

# 3. 두 결과 비교
#    This writes the range comparison plus separate Plain and Backpack grids for all recorded signals.
#    No separate fourth command is needed.
python3 metrics/compare_torque_ranges.py \
  --run-root "$RUN_ROOT" \
  --label-a Plain-G1-original-policy \
  --label-b Backpack-1p30kg-policy
```

For example, a container output directory named
`/workspace/unitree_rl_lab/metrics/shared_runs/plain_original_vs_backpack_1p30kg_20260727_042255`
is available on the host at:

```text
/home/hslee/IsaacLab_ws/unitree_rl_lab/metrics/shared_runs/plain_original_vs_backpack_1p30kg_20260727_042255
```
