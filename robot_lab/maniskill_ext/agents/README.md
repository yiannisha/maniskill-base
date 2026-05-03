# Custom Agents

No custom robots are implemented yet.

When you add one, prefer a ManiSkill Agent that defines:

- `uid`
- URDF or MJCF path
- keyframes
- controller configs
- mounted sensors

After adding the module, import it from `robot_lab/maniskill_ext/agents/__init__.py` and reference the UID from config:

```yaml
env:
  robot_uids: my_robot_uid
```
