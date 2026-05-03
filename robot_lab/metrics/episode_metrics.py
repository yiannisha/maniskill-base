from __future__ import annotations

from collections.abc import Iterable

from robot_lab.runtime.types import EpisodeSummary, RunSummary


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _rate(values: list[bool | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    true_count = sum(1 for value in filtered if value)
    return float(true_count / len(filtered))


def aggregate_episode_summaries(
    experiment_name: str,
    episode_summaries: Iterable[EpisodeSummary],
) -> RunSummary:
    episodes = list(episode_summaries)
    lengths = [float(episode.length) for episode in episodes]
    total_rewards = [float(episode.total_reward) for episode in episodes]
    latencies = [
        float(episode.mean_policy_latency_ms)
        for episode in episodes
        if episode.mean_policy_latency_ms is not None
    ]

    return RunSummary(
        experiment_name=experiment_name,
        num_episodes=len(episodes),
        mean_episode_length=_mean(lengths) or 0.0,
        mean_total_reward=_mean(total_rewards) or 0.0,
        success_rate=_rate([episode.final_success for episode in episodes]),
        success_once_rate=_rate([episode.success_once for episode in episodes]),
        fail_once_rate=_rate([episode.fail_once for episode in episodes]),
        mean_policy_latency_ms=_mean(latencies),
        episodes=[episode.as_dict() for episode in episodes],
    )
