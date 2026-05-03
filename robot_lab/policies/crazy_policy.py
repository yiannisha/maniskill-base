from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from robot_lab.policies.base import Policy
from robot_lab.runtime.types import ActionResult


class CrazyPolicy(Policy):
    """Deliberately chaotic policy for stress-testing rollouts."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params=params)
        self.oscillation_scale = float(self.params.get("oscillation_scale", 0.85))
        self.impulse_prob = float(self.params.get("impulse_prob", 0.35))
        self.flip_prob = float(self.params.get("flip_prob", 0.2))
        self._last_action: np.ndarray | None = None

    def reset(self, episode_context: dict) -> None:
        del episode_context
        self._last_action = None

    def act(
        self,
        obs: Any,
        info: dict,
        action_space: gym.Space,
        t: int,
    ) -> ActionResult:
        del obs, info

        if not isinstance(action_space, gym.spaces.Box):
            return ActionResult(
                action=action_space.sample(),
                debug={"note": "Crazy policy fell back to sampling for non-Box action space."},
            )

        low = np.asarray(action_space.low, dtype=np.float32)
        high = np.asarray(action_space.high, dtype=np.float32)
        center = (low + high) / 2.0
        half_range = (high - low) / 2.0

        phase = np.linspace(0.0, np.pi * 4.0, num=int(np.prod(action_space.shape)), dtype=np.float32)
        phase = phase.reshape(action_space.shape)
        oscillation = np.sin(phase + t * 1.7).astype(np.float32, copy=False)
        action = center + half_range * self.oscillation_scale * oscillation

        if np.random.rand() < self.impulse_prob:
            impulse = np.random.choice([-1.0, 1.0], size=action_space.shape).astype(np.float32, copy=False)
            action = center + half_range * impulse

        if self._last_action is not None and np.random.rand() < self.flip_prob:
            mirrored = center - (self._last_action - center)
            action = 0.5 * action + 0.5 * mirrored

        noise = np.random.uniform(low=-0.25, high=0.25, size=action_space.shape).astype(np.float32, copy=False)
        action = action + noise * half_range
        action = np.clip(action, low, high).astype(action_space.dtype, copy=False)
        self._last_action = np.asarray(action, dtype=np.float32)

        return ActionResult(
            action=action,
            debug={
                "note": "Crazy policy produced a chaotic oscillation/impulse action.",
                "timestep": t,
            },
        )
