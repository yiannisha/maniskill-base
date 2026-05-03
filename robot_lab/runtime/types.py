from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ActionResult:
    action: Any
    command: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None


@dataclass
class EpisodeSummary:
    episode_index: int
    episode_dir: str
    length: int
    total_reward: float
    final_success: bool | None = None
    success_once: bool | None = None
    fail_once: bool | None = None
    mean_policy_latency_ms: float | None = None
    final_info: dict[str, Any] = field(default_factory=dict)
    terminated: bool | None = None
    truncated: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunSummary:
    experiment_name: str
    num_episodes: int
    mean_episode_length: float
    mean_total_reward: float
    success_rate: float | None = None
    success_once_rate: float | None = None
    fail_once_rate: float | None = None
    mean_policy_latency_ms: float | None = None
    episodes: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
