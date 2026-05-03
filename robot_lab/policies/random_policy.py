from __future__ import annotations

from typing import Any

import gymnasium as gym

from robot_lab.policies.base import Policy
from robot_lab.runtime.types import ActionResult


class RandomPolicy(Policy):
    def act(
        self,
        obs: Any,
        info: dict,
        action_space: gym.Space,
        t: int,
    ) -> ActionResult:
        del obs, info, t
        return ActionResult(action=action_space.sample())
