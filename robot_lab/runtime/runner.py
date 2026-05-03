from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from robot_lab.metrics.episode_metrics import aggregate_episode_summaries
from robot_lab.runtime.types import EpisodeSummary
from robot_lab.utils.io import dump_json

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def to_python_scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if torch is not None and torch.is_tensor(value):
        flat = value.detach().cpu().reshape(-1)
        return flat[0].item() if flat.numel() > 0 else default
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        return flat[0].item() if flat.size > 0 else default
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return to_python_scalar(value[0], default=default) if value else default
    return value


def _any_true(value: Any) -> bool:
    if value is None:
        return False
    if torch is not None and torch.is_tensor(value):
        return bool(value.detach().bool().any().cpu().item())
    if isinstance(value, np.ndarray):
        return bool(np.asarray(value).astype(bool).any())
    if isinstance(value, np.generic):
        return bool(value.item())
    if isinstance(value, (list, tuple)):
        return any(_any_true(item) for item in value)
    return bool(value)


def any_done(terminated: Any, truncated: Any) -> bool:
    return _any_true(terminated) or _any_true(truncated)


class RolloutRunner:
    def __init__(
        self,
        cfg: dict,
        env: Any,
        policy: Any,
        evaluator: Any,
        recorder: Any,
        action_adapter: Any,
        observation_adapter: Any,
    ) -> None:
        self.cfg = cfg
        self.env = env
        self.policy = policy
        self.evaluator = evaluator
        self.recorder = recorder
        self.action_adapter = action_adapter
        self.observation_adapter = observation_adapter
        self.output_dir = Path(cfg["experiment"]["output_dir"])
        self.max_episode_steps = cfg["env"].get("max_episode_steps")

    def run(self) -> dict:
        num_episodes = int(self.cfg.get("evaluation", {}).get("num_episodes", 1))
        base_seed = self.cfg.get("experiment", {}).get("seed")
        experiment_name = self.cfg["experiment"]["name"]
        episode_summaries: list[EpisodeSummary] = []

        for episode_index in range(num_episodes):
            episode_seed = None if base_seed is None else int(base_seed) + episode_index
            obs, info = self.env.reset(seed=episode_seed)
            self.env.render()
            episode_dir = self.recorder.start_episode(episode_index, episode_seed=episode_seed)
            self.evaluator.reset()
            self.policy.reset(
                {
                    "episode_index": episode_index,
                    "episode_seed": episode_seed,
                    "env_id": self.cfg["env"]["id"],
                }
            )

            t = 0
            while True:
                policy_obs = self.observation_adapter.adapt(obs)
                t0 = time.perf_counter()
                action_result = self.policy.act(policy_obs, info, self.env.action_space, t)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if action_result.latency_ms is None:
                    action_result.latency_ms = elapsed_ms

                action_result.action = self.action_adapter.adapt(
                    action_result.action,
                    self.env.action_space,
                )

                obs, reward, terminated, truncated, info = self.env.step(action_result.action)
                self.env.render()
                self.evaluator.update(obs, reward, terminated, truncated, info, action_result)
                self.recorder.record_step(t, reward, terminated, truncated, info, action_result)
                t += 1

                if any_done(terminated, truncated):
                    break
                if self.max_episode_steps is not None and t >= int(self.max_episode_steps):
                    break

            episode_summary = self.evaluator.summary(episode_index, episode_dir)
            self.recorder.finalize_episode()
            self.recorder.write_episode_metrics(episode_summary.as_dict())
            episode_summaries.append(episode_summary)

        run_summary = aggregate_episode_summaries(experiment_name, episode_summaries)
        dump_json(run_summary.as_dict(), self.output_dir / "summary.json")
        return run_summary.as_dict()
