from __future__ import annotations

from pathlib import Path
from typing import Any

from robot_lab.runtime.types import ActionResult
from robot_lab.utils.io import dump_json, dump_jsonl, ensure_dir, to_jsonable


class EpisodeRecorder:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.episode_dir: Path | None = None
        self._records: list[dict[str, Any]] = []

    def start_episode(self, episode_index: int, episode_seed: int | None = None) -> Path:
        self.episode_dir = ensure_dir(self.output_dir / f"episode_{episode_index:03d}")
        self._records = []
        metadata = {
            "episode_index": episode_index,
            "episode_seed": episode_seed,
        }
        dump_json(metadata, self.episode_dir / "episode_metadata.json")
        return self.episode_dir

    def record_step(
        self,
        t: int,
        reward: Any,
        terminated: Any,
        truncated: Any,
        info: dict,
        action_result: ActionResult,
    ) -> None:
        record = {
            "t": t,
            "reward": to_jsonable(reward),
            "terminated": to_jsonable(terminated),
            "truncated": to_jsonable(truncated),
            "info_keys": sorted(info.keys()) if isinstance(info, dict) else [],
            "policy_latency_ms": action_result.latency_ms,
            "command": to_jsonable(action_result.command),
            "debug": to_jsonable(action_result.debug),
        }
        self._records.append(record)

    def finalize_episode(self) -> None:
        if self.episode_dir is None:
            raise RuntimeError("start_episode() must be called before finalize_episode().")
        dump_jsonl(self._records, self.episode_dir / "steps.jsonl")

    def write_episode_metrics(self, episode_summary: dict[str, Any]) -> None:
        if self.episode_dir is None:
            raise RuntimeError("start_episode() must be called before write_episode_metrics().")
        dump_json(episode_summary, self.episode_dir / "episode_metrics.json")
