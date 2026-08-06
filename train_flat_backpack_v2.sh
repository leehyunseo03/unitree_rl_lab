#!/usr/bin/env bash
# Reward-shaped flat-backpack training: velocity tracking emphasized,
# torso sway damped.  See velocity_env_cfg_flat_backpack_v2.py for the deltas.
set -euo pipefail
cd "$(dirname "$0")"

/isaac-sim/python.sh scripts/rsl_rl/train.py --headless \
  --task Unitree-G1-29dof-Velocity-Flat-Backpack-V2 \
  --num_envs "${NUM_ENVS:-4096}" \
  --max_iterations "${MAX_ITERATIONS:-50000}" \
  --seed "${SEED:-42}" \
  --logger wandb \
  --log_project_name unitree_rl_lab \
  --run_name "${RUN_NAME:-flat_backpack_v2}" \
  "$@"
