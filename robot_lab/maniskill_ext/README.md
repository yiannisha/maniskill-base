# ManiSkill Extensions

This package is intentionally empty in the first pass.

Use it when you need to add:

- custom ManiSkill tasks
- custom robot agents or form factors
- task-specific registration code

Guideline:

- Put custom task code in `tasks/`.
- Put custom robot agent code in `agents/`.
- Import new modules from the relevant `__init__.py` so registration side effects happen before `gym.make(...)`.
