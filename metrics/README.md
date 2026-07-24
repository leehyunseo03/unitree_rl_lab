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
  --policy /workspace/unitree_rl_lab/container_runs/rsl_rl/unitree_g1_29dof_velocity_backpack/2026-07-13_13-26-45_g1_29dof_velocity_backpack_2kg_dr_pm02_reward_v2_yaw04/exported/policy.onnx \
  --lin-vel-x 0.4 \
  --duration 20 \
  --real-time \
  --livestream 2
```

Use `--duration` to control the recorded simulated seconds. Startup transients are skipped by `--warmup-steps 50` by default.

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
  --run-a 20260720_131153_2026-07-13_13-26-45_g1_29dof_velocity_backpack_2kg_dr_pm02_reward_v2_yaw04_vx0p40_vy0p00_wz0p00 \
  --run-b 20260720_133715_unitree-g1-29dof-velocity_2026-07-13_13-26-45_g1_29dof_velocity_backpack_2kg_dr_pm02_reward_v2_yaw04_vx0p40_vy0p00_wz0p00 \
  --label-a Backpack \
  --label-b Plain-G1
```

The comparison output includes:

- `*_torque_range_compare.png`: blue and red min-to-max torque ranges on the same joint axis.
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
