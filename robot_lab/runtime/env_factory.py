from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401

import robot_lab.maniskill_ext  # noqa: F401


def make_env(cfg: dict) -> gym.Env:
    env_cfg = cfg["env"]
    recording_cfg = cfg.get("recording", {})

    kwargs: dict[str, object] = {}
    for key in (
        "num_envs",
        "obs_mode",
        "control_mode",
        "render_mode",
        "robot_uids",
        "max_episode_steps",
    ):
        value = env_cfg.get(key)
        if value is not None:
            kwargs[key] = value

    if recording_cfg.get("enabled", False) and recording_cfg.get("save_video", False):
        if kwargs.get("render_mode") == "human":
            kwargs["render_mode"] = "rgb_array"

    kwargs.update(env_cfg.get("kwargs", {}))

    env = gym.make(env_cfg["id"], **kwargs)

    if recording_cfg.get("enabled", False):
        from mani_skill.utils.wrappers.record import RecordEpisode

        print(f"Wrapping env with RecordEpisode wrapper for recording. Recording config: {recording_cfg}")

        output_dir = Path(cfg["experiment"]["output_dir"])
        env = RecordEpisode(
            env,
            output_dir=str(output_dir),
            save_video=bool(recording_cfg.get("save_video", False)),
            save_trajectory=bool(recording_cfg.get("save_trajectory", False)),
            video_fps=int(recording_cfg.get("video_fps", 30)),
            trajectory_name="trajectory",
        )

    return env
