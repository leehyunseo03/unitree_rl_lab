# Train
```
./unitree_rl_lab.sh -t \
  --task Unitree-G1-29dof-Velocity-Flat-Backpack \
  --num_envs 4096 \
  --max_iterations 50000 \
  --logger wandb \
  --log_project_name unitree_rl_lab \
  --run_name flat_backpack_v1
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

