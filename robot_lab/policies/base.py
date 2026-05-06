from __future__ import annotations

from typing import Any

import gymnasium as gym

from robot_lab.runtime.types import ActionResult


class Policy:
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def get_reset_kwargs(self, episode_context: dict[str, Any]) -> dict[str, Any]:
        del episode_context
        return {}

    def reset(self, episode_context: dict) -> None:
        del episode_context

    def act(
        self,
        obs: Any,
        info: dict,
        action_space: gym.Space,
        t: int,
    ) -> ActionResult:
        raise NotImplementedError
