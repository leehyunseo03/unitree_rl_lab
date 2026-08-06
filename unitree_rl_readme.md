## Conver pt to onnx
```
/isaac-sim/python.sh scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Velocity-Backpack \
  --checkpoint /workspace/unitree_rl_lab/container_runs/rsl_rl/unitree_g1_29dof_velocity_backpack/2026-08-04_13-05-24_backpack_v2_ft50k/model_123900.pt \
  --num_envs 1 \
  --headless \
  --export-only
```

/home/hslee/IsaacLab_ws/unitree_rl_lab/container_runs/rsl_rl/unitree_g1_29dof_velocity_backpack/2026-07-25_18-25-54_g1_29dof_velocity_backpack_1p30kg_dr_pm02_y4cm_z5cm_omni_floor_dr_stable_resume_to_100k

## Policy Test
```
LIVESTREAM=2 /isaac-sim/python.sh scripts/rsl_rl/play_onnx.py \
    --task Unitree-G1-29dof-Velocity-Backpack \
    --policy /workspace/unitree_rl_lab/container_runs/rsl_rl/unitree_g1_29dof_velocity_backpack/2026-08-05_13-51-00_backpack_scratch_50k/exported/policy.onnx \
    --command -0.3 0.0 0.0 \
    --real-time \
    --livestream 2 \
    --no-camera-follow
```



/home/hslee/IsaacLab_ws/unitree_rl_lab/container_runs/rsl_rl/unitree_g1_29dof_velocity_backpack/2026-08-05_13-51-00_backpack_scratch_50k

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
  --run_name g1_29dof_velocity_backpack_1p30kg_dr_pm02_y4cm_z5cm_omni_50k
```

```
/isaac-sim/python.sh scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Velocity-Backpack \
  --headless \
  --num_envs 4096 \
  --max_iterations 50000 \
  --log_root_path /workspace/unitree_rl_lab/container_runs/rsl_rl \
  --logger wandb \
  --log_project_name unitree_rl_lab \
  --run_name g1_29dof_velocity_backpack_1p30kg_dr_pm02_y4cm_z5cm_omni_50k
```
## Train Contiuously
```
cd /workspace/unitree_rl_lab

/isaac-sim/python.sh scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Velocity-Backpack \
  --headless \
  --num_envs 4096 \
  --max_iterations 43000 \
  --log_root_path /workspace/unitree_rl_lab/container_runs/rsl_rl \
  --resume \
  --load_run 2026-07-22_07-29-31_g1_29dof_velocity_backpack_1p5kg_dr_pm02_50k_resume \
  --checkpoint model_7100.pt \
  --logger wandb \
  --log_project_name unitree_rl_lab \
  --run_name g1_29dof_velocity_backpack_1p5kg_dr_pm02_50k_resume
```

/isaac-sim/python.sh scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Velocity-Backpack \
  --headless \
  --num_envs 4096 \
  --resume \
  --load_run 2026-07-24_13-32-33_g1_29dof_velocity_backpack_1p30kg_dr_pm02_y4cm_z5cm_omni_50k \
  --checkpoint model_49999.pt \
  --max_iterations 50000 \
  --log_root_path /workspace/unitree_rl_lab/container_runs/rsl_rl \
  --logger wandb \
  --log_project_name unitree_rl_lab \
  --run_name g1_29dof_velocity_backpack_1p30kg_dr_pm02_y4cm_z5cm_omni_floor_dr_stable_20k


## Train from scratch
```
/isaac-sim/python.sh scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Velocity-Backpack \
  --headless \
  --num_envs 4096 \
  --max_iterations 50000 \
  --log_root_path /workspace/unitree_rl_lab/container_runs/rsl_rl \
  --logger wandb \
  --log_project_name unitree_rl_lab \
  --run_name backpack_scratch_50k
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