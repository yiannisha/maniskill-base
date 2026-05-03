# Examples

Use the configs in `configs/experiments/` as the first examples:

- `pick_cube_random_state.yaml`
- `pick_cube_random_rgbd_record.yaml`
- `pick_cube_scripted_state.yaml`
- `pick_cube_remote_policy_stub.yaml`

Recommended order:

1. `python scripts/inspect_env.py --env-id PickCube-v1 --obs-mode state --control-mode pd_joint_delta_pos`
2. `python scripts/run_eval.py --config configs/experiments/pick_cube_random_state.yaml`
3. `python scripts/collect_rollouts.py --config configs/experiments/pick_cube_random_rgbd_record.yaml`
