## Conver pt to onnx
```
/isaac-sim/python.sh scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Velocity-Backpack \
  --checkpoint /workspace/unitree_rl_lab/container_runs/rsl_rl/unitree_g1_29dof_velocity_backpack/2026-07-13_13-26-45_g1_29dof_velocity_backpack_2kg_dr_pm02_reward_v2_yaw04/model_49999.pt \
  --num_envs 1 \
  --headless \
  --export-only
```

## Policy Test
```
LIVESTREAM=2 /isaac-sim/python.sh scripts/rsl_rl/play_onnx.py \
    --task Unitree-G1-29dof-Velocity-Backpack \
    --policy /workspace/unitree_rl_lab/container_runs/rsl_rl/unitree_g1_29dof_velocity_backpack/2026-07-13_13-26-45_g1_29dof_velocity_backpack_2kg_dr_pm02_reward_v2_yaw04/exported/policy.onnx \
    --lin-vel-x 0.4 \
    --real-time \
    --livestream 2
```

## Train
```
python scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Velocity-Backpack \
  --headless \
  --num_envs 4096 \
  --max_iterations 50000
  
```
```
python scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Velocity-Backpack \
  --headless \
  --num_envs 4096 \
  --max_iterations 50000 \
  --logger wandb \
  --log_project_name unitree_rl_lab \
  --run_name g1_29dof_velocity_backpack_2kg_50k
```

## sim2sim (IsaacSim -> Mujoco)
#### Mujoco
```
conda activate unitree-mujoco

cd /home/hslee/IsaacLab_ws/unitree_mujoco/simulate/build

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/home/hslee/IsaacLab_ws/.local/unitree_sdk2/lib:../mujoco/lib:$LD_LIBRARY_PATH"

./unitree_mujoco -r g1 -s scene_29dof.xml
```
#### Controller
```
cd /home/hslee/IsaacLab_ws/unitree_rl_lab/deploy/robots/g1_29dof/build

export LD_LIBRARY_PATH="/home/hslee/IsaacLab_ws/.local/unitree_sdk2/lib:$LD_LIBRARY_PATH"

./g1_ctrl
```