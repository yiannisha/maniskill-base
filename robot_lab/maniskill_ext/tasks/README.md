# Custom Tasks

No custom tasks are implemented yet.

When you add one, prefer a ManiSkill task registered with `register_env` that defines:

- scene assets
- reset/randomization
- `evaluate()` success and failure signals
- optional sensors and cameras
- optional dense rewards if useful

After adding the module, import it from `robot_lab/maniskill_ext/tasks/__init__.py` and reference the task ID from config:

```yaml
env:
  id: MyTask-v0
```
