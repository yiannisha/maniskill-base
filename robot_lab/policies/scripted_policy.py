from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from robot_lab.policies.base import Policy
from robot_lab.runtime.types import ActionResult


class ScriptedPolicy(Policy):
    """Small-action placeholder heuristic.

    This is intentionally not task-specific and is not expected to solve PickCube
    robustly. It only demonstrates how a non-random policy plugs into the runner.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params=params)
        self.action_scale = float(self.params.get("action_scale", 0.05))

    def act(
        self,
        obs: Any,
        info: dict,
        action_space: gym.Space,
        t: int,
    ) -> ActionResult:
        del obs, info, t

        if isinstance(action_space, gym.spaces.Box):
            action = np.random.uniform(
                low=-self.action_scale,
                high=self.action_scale,
                size=action_space.shape,
            ).astype(action_space.dtype, copy=False)
        else:
            action = action_space.sample()

        return ActionResult(
            action=action,
            debug={"note": "Placeholder small-magnitude heuristic action."},
        )
