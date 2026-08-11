# Train
```
/isaac-sim/python.sh scripts/rsl_rl/train.py --headless \
  --task Unitree-G1-29dof-Velocity-Flat-Backpack \
  --num_envs 4096 \
  --max_iterations 50000 \
  --logger wandb \
  --log_project_name unitree_rl_lab \
  --run_name flat_backpack_v1
```

```
/isaac-sim/python.sh scripts/rsl_rl/train.py --headless \
  --task Unitree-G1-29dof-Velocity-Flat-Backpack-V2 \
  --num_envs 4096 \
  --max_iterations 50000 \
  --logger wandb \
  --log_project_name unitree_rl_lab \
  --run_name flat_backpack_v2
```

Torso pitch/yaw stability reward (v3):

```
/isaac-sim/python.sh scripts/rsl_rl/train.py --headless \
  --task Unitree-G1-29dof-Velocity-Flat-Backpack-V3 \
  --num_envs 4096 \
  --max_iterations 50000 \
  --logger wandb \
  --log_project_name unitree_rl_lab \
  --run_name flat_backpack_v3
```

```
/isaac-sim/python.sh scripts/rsl_rl/train.py --headless \
  --task Unitree-G1-29dof-Velocity-Flat-Backpack-Stand \
  --num_envs 4096 \
  --max_iterations 50000 \
  --logger wandb \
  --log_project_name unitree_rl_lab \
  --run_name flat_backpack_stand_v1
```

### velocity_curriculum v2
```
/isaac-sim/python.sh scripts/rsl_rl/train.py --headless \
  --task Unitree-G1-29dof-Velocity-Flat-Backpack-Unitree \
  --num_envs 4096 \
  --max_iterations 50000 \
  --logger wandb \
  --log_project_name unitree_rl_lab \
  --run_name flat_backpack_unitree_v2
```
- Model path : /home/hslee/IsaacLab_ws/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity_flat_backpack/2026-08-06_18-30-00_flat_backpack_v1/model_30000.pt

### velocity_curriculum v3

# Convert into onnx
```
/isaac-sim/python.sh scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Velocity-Flat-Backpack \
  --checkpoint /workspace/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity_backpack/stand/model_16300.pt \
  --num_envs 1 \
  --headless 
```

# Test
```
LIVESTREAM=2 /isaac-sim/python.sh scripts/rsl_rl/play_onnx.py \
    --task Unitree-G1-29dof-Velocity-Flat-Backpack \
    --policy /workspace/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/basic/policy.onnx \
    --command 0.3 0.0 0.0 \
    --real-time \
    --livestream 2 
```
