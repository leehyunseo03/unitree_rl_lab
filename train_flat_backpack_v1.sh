#!/usr/bin/env bash
# Baseline flat-backpack training (original reward weights).
set -euo pipefail
cd "$(dirname "$0")"

/isaac-sim/python.sh scripts/rsl_rl/train.py --headless \
  --task Unitree-G1-29dof-Velocity-Flat-Backpack \
  --num_envs "${NUM_ENVS:-4096}" \
  --max_iterations "${MAX_ITERATIONS:-50000}" \
  --seed "${SEED:-42}" \
  --logger wandb \
  --log_project_name unitree_rl_lab \
  --run_name "${RUN_NAME:-flat_backpack_v1}" \
  "$@"
