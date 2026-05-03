from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


class ActionAdapter:
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def adapt(self, action: Any, action_space: gym.Space) -> Any:
        del action_space
        return action


class IdentityActionAdapter(ActionAdapter):
    pass


class ClipBoxActionAdapter(ActionAdapter):
    def adapt(self, action: Any, action_space: gym.Space) -> Any:
        if not isinstance(action_space, gym.spaces.Box):
            return action

        if torch is not None and torch.is_tensor(action):
            low = torch.as_tensor(action_space.low, dtype=action.dtype, device=action.device)
            high = torch.as_tensor(action_space.high, dtype=action.dtype, device=action.device)
            return torch.clamp(action, min=low, max=high)

        clipped = np.clip(np.asarray(action), action_space.low, action_space.high)
        return clipped.astype(action_space.dtype, copy=False)
