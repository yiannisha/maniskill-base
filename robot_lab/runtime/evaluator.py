from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robot_lab.runtime.types import ActionResult, EpisodeSummary
from robot_lab.utils.io import to_jsonable

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def _first_scalar(value: Any) -> Any:
    if value is None:
        return None
    if torch is not None and torch.is_tensor(value):
        flat = value.detach().cpu().reshape(-1)
        return flat[0].item() if flat.numel() > 0 else None
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        return flat[0].item() if flat.size > 0 else None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return _first_scalar(value[0]) if value else None
    return value


def _to_bool(value: Any) -> bool | None:
    scalar = _first_scalar(value)
    if scalar is None:
        return None
    if isinstance(scalar, bool):
        return scalar
    if isinstance(scalar, (int, float)):
        return bool(scalar)
    return None


def _to_float(value: Any) -> float:
    scalar = _first_scalar(value)
    if scalar is None:
        return 0.0
    if isinstance(scalar, (bool, int, float)):
        return float(scalar)
    return 0.0


class EpisodeEvaluator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.length = 0
        self.total_reward = 0.0
        self.final_success: bool | None = None
        self.success_once: bool | None = None
        self.fail_once: bool | None = None
        self.terminated: bool | None = None
        self.truncated: bool | None = None
        self.final_info: dict[str, Any] = {}
        self._policy_latencies_ms: list[float] = []

    def update(
        self,
        obs: Any,
        reward: Any,
        terminated: Any,
        truncated: Any,
        info: dict,
        action_result: ActionResult,
    ) -> None:
        del obs
        self.length += 1
        self.total_reward += _to_float(reward)
        self.terminated = _to_bool(terminated)
        self.truncated = _to_bool(truncated)

        success = _to_bool(info.get("success")) if isinstance(info, dict) else None
        fail = _to_bool(info.get("fail")) if isinstance(info, dict) else None

        if success is not None:
            self.final_success = success
            self.success_once = success if self.success_once is None else self.success_once or success
        if fail is not None:
            self.fail_once = fail if self.fail_once is None else self.fail_once or fail

        if isinstance(info, dict):
            self.final_info = to_jsonable(info)

        if action_result.latency_ms is not None:
            self._policy_latencies_ms.append(float(action_result.latency_ms))

    def summary(self, episode_index: int, episode_dir: Path) -> EpisodeSummary:
        mean_latency = None
        if self._policy_latencies_ms:
            mean_latency = float(sum(self._policy_latencies_ms) / len(self._policy_latencies_ms))

        return EpisodeSummary(
            episode_index=episode_index,
            episode_dir=str(episode_dir),
            length=self.length,
            total_reward=float(self.total_reward),
            final_success=self.final_success,
            success_once=self.success_once,
            fail_once=self.fail_once,
            mean_policy_latency_ms=mean_latency,
            final_info=self.final_info,
            terminated=self.terminated,
            truncated=self.truncated,
        )
