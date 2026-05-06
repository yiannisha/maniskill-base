# AGENTS.md

This repo is a small ManiSkill evaluation harness. The main "tools" exposed by the codebase are:

- runnable scripts in `scripts/`
- policy implementations in `robot_lab/policies/`
- action and observation adapters in `robot_lab/adapters/`
- custom ManiSkill environments in `robot_lab/maniskill_ext/tasks/`

## Scripts

### `scripts/run_eval.py`

Primary evaluation entry point.

- Reads an experiment YAML via `--config`
- Builds the environment, policy, adapters, evaluator, and recorder
- Runs closed-loop rollouts
- Writes outputs under `experiment.output_dir`

Use this when you want normal evaluation runs and summary metrics.

Example:

```bash
python scripts/run_eval.py --config configs/experiments/pick_cube_random_state.yaml
```

### `scripts/collect_rollouts.py`

Rollout collection entry point.

- Same core execution path as `run_eval.py`
- Intended for recording data
- Supports `--force-recording` to turn on recording even if the config disables it

Use this when you want to guarantee trajectory and/or video capture.

Example:

```bash
python scripts/collect_rollouts.py \
  --config configs/experiments/pick_cube_random_state.yaml \
  --force-recording
```

### `scripts/inspect_env.py`

Quick environment probe.

- Instantiates a ManiSkill env directly
- Prints observation space and action space
- Runs one reset and one sampled step

Use this first when debugging env ids, observation modes, control modes, or rendering issues.

Example:

```bash
python scripts/inspect_env.py \
  --env-id PickCube-v1 \
  --obs-mode state \
  --control-mode pd_joint_delta_pos
```

### `scripts/extract_trajectory_h5.py`

Trajectory extraction utility.

- Reads a ManiSkill `trajectory.h5`
- Creates a readable folder tree next to it by default
- Saves every dataset as `.npy`
- Exports RGB streams as PNG frame folders
- Exports depth as raw `.npy` plus preview PNGs

Use this when VS Code or Finder is not enough for inspecting trajectory data.

Example:

```bash
python scripts/extract_trajectory_h5.py \
  --experiment-dir runs/pick_cube_third_person_rgbd_record \
  --overwrite
```

## Policies

Policies are selected with:

```yaml
policy:
  type: ...
  params: {}
```

Registered policy types in `robot_lab/policies/__init__.py`:

### `random`

Implemented by `RandomPolicy`.

- Samples directly from the action space
- Good for smoke tests

### `scripted`

Implemented by `ScriptedPolicy`.

- Emits small random actions
- Not task-solving logic
- Useful as a non-zero baseline or integration stub

Key params:

- `action_scale`

### `crazy`

Implemented by `CrazyPolicy`.

- Produces oscillatory, impulsive, high-variance actions
- Useful for stress-testing rollout and recorder behavior

Key params:

- `oscillation_scale`
- `impulse_prob`
- `flip_prob`

### `demo_replay`

Implemented by `DemoReplayPolicy`.

- Replays actions from a ManiSkill demo dataset or explicit trajectory file
- Aligns episodes with demo metadata
- Can reject mismatched control modes or env ids

Key params:

- `env_id` or `demo_env_id`
- `source`
- `trajectory_path`
- `demo_root`
- `control_mode`
- `cycle_episodes`
- `strict_env_id_match`

### `remote_stub`

Implemented by `RemotePolicyStub`.

- Placeholder for HTTP/gRPC/remote inference integration
- Does not implement transport by default
- Can optionally fall back to random actions

Key params:

- `endpoint`
- `timeout_s`
- `fallback_to_random`

### `openpi`

Implemented by `OpenPIPolicy`.

- Runs a Physical Intelligence OpenPI policy from this harness
- Supports local in-process inference via OpenPI's `create_trained_policy`
- Also supports websocket inference against an OpenPI policy server
- Maps ManiSkill observations into configurable OpenPI observation keys
- Consumes OpenPI action chunks with a configurable open-loop horizon

Key params:

- `mode`
- `config_name`
- `checkpoint_dir`
- `prompt`
- `observation`
- `open_loop_horizon`
- `action_key`

## Action Adapters

Action adapters are selected with:

```yaml
action_adapter:
  type: ...
  params: {}
```

Available action adapters in `robot_lab/adapters/__init__.py`:

### `identity`

- Returns the action unchanged

### `clip_box`

- Clips numeric actions to the bounds of a `gym.spaces.Box`
- Supports NumPy arrays and Torch tensors

Use `clip_box` when your policy may overshoot actuator limits.

## Observation Adapters

Observation adapters are selected with:

```yaml
observation_adapter:
  type: ...
  params: {}
```

Available observation adapters:

### `identity`

- Passes observations through unchanged

### `flatten_state`

- If the observation is a dict containing `state`, returns `obs["state"]`
- Otherwise passes the observation through unchanged

Use this when the env returns a structured dict but your policy only wants the flattened state vector.

## Custom Environments

Custom envs are auto-imported through `robot_lab.maniskill_ext`.

### `PickCubeThirdPersonCamera-v0`

Defined in `robot_lab/maniskill_ext/tasks/pick_cube_third_person_camera.py`.

- Extends ManiSkill `PickCubeEnv`
- Keeps the default `base_camera`
- Adds a fixed `third_person_camera`

When `obs_mode: rgbd` and trajectory recording are enabled, both camera streams are saved into `trajectory.h5` under:

- `traj_N/obs/sensor_data/base_camera/...`
- `traj_N/obs/sensor_data/third_person_camera/...`

### `PickCubeWithGuy-v0`

Defined in `robot_lab/maniskill_ext/tasks/pick_cube_with_guy.py`.

- Extends ManiSkill `PickCubeEnv`
- Loads the `the_guy` URDF as a fixed scene articulation
- Requires `the_guy/the_guy.urdf` to exist in the repo root

## Runtime Components

These are not configured as top-level "tools" by name, but they are the core execution pieces:

### `make_env`

Defined in `robot_lab/runtime/env_factory.py`.

- Instantiates the ManiSkill env from YAML
- Wraps it with ManiSkill `RecordEpisode` when recording is enabled

### `RolloutRunner`

Defined in `robot_lab/runtime/runner.py`.

- Executes the episode loop
- Calls `policy.get_reset_kwargs()`, `policy.reset()`, and `policy.act()`
- Applies adapters
- Updates evaluation and recording

### `EpisodeRecorder`

Defined in `robot_lab/runtime/recorder.py`.

- Writes repo-side per-episode sidecar files like:
  - `episode_000/steps.jsonl`
  - `episode_000/episode_metrics.json`

Note that video files and `trajectory.h5` are written by ManiSkill's `RecordEpisode`, not by this recorder.

## Outputs

Typical run outputs under `runs/<experiment_name>/`:

```text
runs/<experiment_name>/
  config.yaml
  summary.json
  trajectory.h5
  trajectory.json
  *.mp4
  episode_000/
    episode_metrics.json
    steps.jsonl
```

Use the outputs like this:

- `summary.json`: run-level metrics
- `episode_*/steps.jsonl`: policy-side step traces
- `trajectory.h5`: simulator-side source of truth
- `*.mp4`: rendered videos

## Config Notes

Most work is driven from YAML files in `configs/experiments/`.

The most important sections are:

- `experiment`
- `env`
- `policy`
- `action_adapter`
- `observation_adapter`
- `recording`
- `evaluation`
- `runner`

If a new tool is added to the repo, it should usually be registered in one of these places:

- a new runnable script under `scripts/`
- a new policy in `robot_lab/policies/__init__.py`
- a new adapter in `robot_lab/adapters/__init__.py`
- a new env in `robot_lab/maniskill_ext/tasks/`
