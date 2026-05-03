# maniskill-robot-lab

`maniskill-robot-lab` is a bare-bones Python repo for running inference-only closed-loop control, evaluation, and rollout collection on top of ManiSkill.

ManiSkill owns:

- simulation
- built-in tasks and scene randomization
- robot agents and low-level controllers
- sensors and observation generation
- trajectory/video recording wrappers

This repo owns:

- YAML experiment configs
- rollout orchestration
- policy and adapter registries
- evaluation summaries
- sidecar logging for policy traces and per-episode metrics

## What This Repo Is

- A minimal evaluation and data-collection harness around ManiSkill.
- A simple place to swap tasks, robots, policies, and control pipelines.
- A starting point for plugging in classical controllers or remote ML inference services.

## What This Repo Is Not

- Not a training repo.
- Not an RL benchmark framework.
- Not an imitation learning stack.
- Not a custom robot/task implementation yet.
- Not tied to any specific VLA or remote model.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If your shell exposes `python3` instead of `python`, substitute accordingly.

ManiSkill installation can vary by OS, CUDA, Vulkan, and renderer support. If `pip install -r requirements.txt` is not enough for your machine, follow the official ManiSkill installation docs for your exact setup:

- https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html
- https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/macos_install.html

Notes:

- State-only tasks are usually the easiest path to first success.
- Visual observation modes such as `rgbd` depend on ManiSkill rendering support on the local machine.
- Some tasks may prompt for asset downloads the first time you run them.

## Running The First Sample

Inspect the environment first:

```bash
python scripts/inspect_env.py --env-id PickCube-v1 --obs-mode state --control-mode pd_joint_delta_pos
```

Run random-policy evaluation on PickCube:

```bash
python scripts/run_eval.py \
  --config configs/experiments/pick_cube_random_state.yaml
```

Run the heuristic placeholder policy:

```bash
python scripts/run_eval.py \
  --config configs/experiments/pick_cube_scripted_state.yaml
```

## Recording Trajectories And Videos

Record one rollout with RGB-D observations and ManiSkill's `RecordEpisode` wrapper:

```bash
python scripts/collect_rollouts.py \
  --config configs/experiments/pick_cube_random_rgbd_record.yaml
```

If you want to force recording from another config:

```bash
python scripts/collect_rollouts.py \
  --config configs/experiments/pick_cube_random_state.yaml \
  --force-recording
```

## Repo Architecture

```text
configs/
  experiments/         YAML experiment definitions

robot_lab/
  runtime/             env creation, rollout runner, evaluation, recording
  policies/            random/scripted/remote policy stubs
  adapters/            observation and action adapters
  metrics/             episode aggregation helpers
  utils/               config, IO, seeding
  maniskill_ext/       placeholders for custom ManiSkill tasks and agents

scripts/
  run_eval.py          main evaluation entry point
  collect_rollouts.py  rollout collection entry point
  inspect_env.py       quick env/action-space inspection
```

## Config Structure

Each experiment is a plain YAML file. No Hydra or schema library is used.

Example:

```yaml
experiment:
  name: pick_cube_random_state
  seed: 0
  output_dir: runs/pick_cube_random_state

env:
  id: PickCube-v1
  num_envs: 1
  obs_mode: state
  control_mode: pd_joint_delta_pos
  render_mode: null
  max_episode_steps: 100
  robot_uids: panda
  kwargs: {}

policy:
  type: random
```

## Data And Output Layout

Runs are written under `runs/<experiment_name>/`:

```text
runs/<experiment_name>/
  config.yaml
  summary.json
  trajectory.h5 / trajectory.json       # when RecordEpisode trajectory saving is enabled
  *.mp4                                 # when RecordEpisode video saving is enabled
  episode_000/
    episode_metrics.json
    steps.jsonl
  episode_001/
    episode_metrics.json
    steps.jsonl
```

ManiSkill trajectory files remain the source of truth for rollout replay and simulator-side data. This repo adds sidecar metrics and policy traces.

## Adding A New Policy

1. Create `robot_lab/policies/my_policy.py`.
2. Subclass `robot_lab.policies.base.Policy`.
3. Implement `reset()` and `act()`.
4. Register it in `robot_lab/policies/__init__.py`.
5. Use it in YAML:

```yaml
policy:
  type: my_policy
  params:
    checkpoint: path/to/model.pt
```

Policy code should stay simulator-light. It only receives:

- `obs`
- `info`
- `action_space`
- timestep `t`

It should return `ActionResult`.

## Adding A New Action Adapter

1. Create a new adapter class in `robot_lab/adapters/action_adapter.py` or a sibling module.
2. Implement `adapt(action, action_space)`.
3. Register it in `robot_lab/adapters/__init__.py`.
4. Reference it in YAML:

```yaml
action_adapter:
  type: my_adapter
  params:
    scale: 0.1
```

Use action adapters when the policy output is not yet a final ManiSkill action. Examples:

- end-effector delta pose
- joint targets
- base velocity
- grasp command
- model outputs that need clipping or decoding

If the change is low-level actuation behavior, prefer implementing it as a ManiSkill controller on the robot agent instead.

## Adding A New Observation Adapter

1. Create a new adapter class in `robot_lab/adapters/observation_adapter.py` or a sibling module.
2. Implement `adapt(obs)`.
3. Register it in `robot_lab/adapters/__init__.py`.
4. Reference it in YAML:

```yaml
observation_adapter:
  type: my_obs_adapter
  params:
    image_key: hand_camera
```

Use observation adapters when a policy wants a different observation view than the raw ManiSkill output.

## Adding A New ManiSkill Task

Prefer implementing the task inside `robot_lab/maniskill_ext/tasks/`.

Suggested flow:

1. Create a ManiSkill task module in `robot_lab/maniskill_ext/tasks/`.
2. Register it with ManiSkill's `register_env`.
3. Define scene assets, reset/randomization, evaluation logic, and optional cameras.
4. Import the module from `robot_lab/maniskill_ext/tasks/__init__.py`.
5. Point config at the new task:

```yaml
env:
  id: MyTask-v0
```

The task should define:

- scene assets
- reset/randomization
- `evaluate()` success and failure metrics
- optional cameras
- optional dense rewards if useful

This repo does not require rewards for training because it does not implement training.

## Adding A New ManiSkill Robot Or Form Factor

Prefer implementing the robot as a ManiSkill Agent inside `robot_lab/maniskill_ext/agents/`.

Suggested flow:

1. Create a new agent module in `robot_lab/maniskill_ext/agents/`.
2. Define its UID, URDF or MJCF, keyframes, controller configs, and mounted sensors.
3. Import the module from `robot_lab/maniskill_ext/agents/__init__.py`.
4. Use it in config:

```yaml
env:
  robot_uids: my_robot_uid
```

This repo intentionally does not ship a custom robot in the first pass.

## Connecting A Remote ML Model

Use a policy class, not the runner, to integrate remote inference.

Recommended flow:

1. Subclass `Policy`.
2. Preprocess `obs` and `info` inside the policy.
3. Call an HTTP, gRPC, or other remote service.
4. Decode the response into a policy output.
5. Return `ActionResult`.
6. Use an `ActionAdapter` to map the model output into the final ManiSkill action when needed.

The included `RemotePolicyStub` shows this boundary without requiring a real server.

Policy code should not depend on ManiSkill internals beyond the observation object, info dict, and action space.

## Extension Flows

### A. Add New Policy

1. Create `robot_lab/policies/my_policy.py`.
2. Subclass `Policy`.
3. Implement `reset` and `act`.
4. Register it in the policy registry.
5. Use it in YAML.

### B. Add VLA Or Remote Inference Model

The policy should:

- preprocess observations
- call a model or remote server
- decode the output
- return `ActionResult`

Use an action adapter to translate the model output into ManiSkill actions.

### C. Add New Task

Prefer adding the task in `robot_lab/maniskill_ext/tasks/`, registering it with `register_env`, then switching `env.id` in YAML.

### D. Add New Form Factor Or Robot

Prefer adding a ManiSkill Agent in `robot_lab/maniskill_ext/agents/`, then switching `env.robot_uids` in YAML.

### E. Add New Control Mode

- If it is low-level actuation, implement it as a ManiSkill controller on the robot agent.
- If it is high-level decision-making, implement it as a `Policy` or `ActionAdapter` in this repo.

### F. Add New Benchmark

Create a YAML config first. Only add code if existing tasks, policies, and adapters cannot express the benchmark cleanly.

### G. Add Dataset Collection

Use `collect_rollouts.py` with recording enabled. Keep ManiSkill trajectory files as the source of truth and store policy traces and metrics as sidecar JSON/JSONL files.

## Known Limitations

- Built for `num_envs=1` first, though the code is structured so `num_envs>1` can be added later.
- No vectorized auto-reset logic yet.
- No custom tasks or custom robots are implemented yet.
- No training loop.
- No distributed rollout infrastructure.
- The scripted policy is only a placeholder heuristic and is not expected to solve PickCube reliably.
- The remote policy is a stub transport boundary, not a real client.
- Rendering and RGB-D collection depend on the local ManiSkill installation and renderer support.
