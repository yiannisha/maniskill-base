from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from robot_lab.policies.base import Policy
from robot_lab.runtime.types import ActionResult
from robot_lab.utils.io import to_jsonable

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


class RemotePolicyStub(Policy):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params=params)
        self.endpoint = str(self.params.get("endpoint", "http://localhost:8000/act"))
        self.timeout_s = float(self.params.get("timeout_s", 5.0))
        self.fallback_to_random = bool(self.params.get("fallback_to_random", False))

    def act(
        self,
        obs: Any,
        info: dict,
        action_space: gym.Space,
        t: int,
    ) -> ActionResult:
        payload = {
            "t": t,
            "obs": self._serialize_obs(obs),
            "info": self._serialize_obs(info),
            "action_space": self._describe_action_space(action_space),
        }

        debug = {"request_preview": payload}
        try:
            action = self._request_action(payload, action_space)
        except NotImplementedError as exc:
            if not self.fallback_to_random:
                raise
            action = action_space.sample()
            debug["fallback_reason"] = str(exc)

        return ActionResult(
            action=action,
            command={"endpoint": self.endpoint, "timeout_s": self.timeout_s},
            debug=debug,
        )

    def _request_action(self, payload: dict[str, Any], action_space: gym.Space) -> Any:
        del payload, action_space
        raise NotImplementedError(
            "RemotePolicyStub does not implement transport. "
            "Replace _request_action() for HTTP/gRPC/etc., or set fallback_to_random: true."
        )

    def _serialize_obs(self, value: Any) -> Any:
        if torch is not None and torch.is_tensor(value):
            return {
                "type": "torch.Tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        if isinstance(value, np.ndarray):
            return {
                "type": "numpy.ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        if isinstance(value, dict):
            return {str(k): self._serialize_obs(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serialize_obs(v) for v in value]
        return to_jsonable(value)

    def _describe_action_space(self, action_space: gym.Space) -> dict[str, Any]:
        description: dict[str, Any] = {
            "type": type(action_space).__name__,
        }
        if isinstance(action_space, gym.spaces.Box):
            description.update(
                {
                    "shape": list(action_space.shape),
                    "dtype": str(action_space.dtype),
                    "low": to_jsonable(action_space.low),
                    "high": to_jsonable(action_space.high),
                }
            )
        return description
