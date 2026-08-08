#!/usr/bin/env bash
# Flat-backpack training with explicit torso pitch/yaw stabilization rewards.
set -euo pipefail
cd "$(dirname "$0")"

/isaac-sim/python.sh scripts/rsl_rl/train.py --headless \
  --task Unitree-G1-29dof-Velocity-Flat-Backpack-V3 \
  --num_envs "${NUM_ENVS:-4096}" \
  --max_iterations "${MAX_ITERATIONS:-50000}" \
  --seed "${SEED:-42}" \
  --logger wandb \
  --log_project_name unitree_rl_lab \
  --run_name "${RUN_NAME:-flat_backpack_v3}" \
  "$@"
