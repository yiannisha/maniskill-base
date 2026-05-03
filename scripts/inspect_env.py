#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import mani_skill.envs  # noqa: F401

import robot_lab.maniskill_ext  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a ManiSkill environment.")
    parser.add_argument("--env-id", required=True, help="Environment ID, e.g. PickCube-v1")
    parser.add_argument("--obs-mode", default="state", help="Observation mode")
    parser.add_argument("--control-mode", default=None, help="Control mode")
    parser.add_argument("--render-mode", default=None, help="Render mode")
    parser.add_argument("--robot-uids", default=None, help="Robot UID or UIDs")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--max-episode-steps", type=int, default=None, help="Optional time limit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    kwargs: dict[str, object] = {
        "num_envs": args.num_envs,
        "obs_mode": args.obs_mode,
    }
    if args.control_mode is not None:
        kwargs["control_mode"] = args.control_mode
    if args.render_mode is not None:
        kwargs["render_mode"] = args.render_mode
    if args.robot_uids is not None:
        kwargs["robot_uids"] = args.robot_uids
    if args.max_episode_steps is not None:
        kwargs["max_episode_steps"] = args.max_episode_steps

    env = gym.make(args.env_id, **kwargs)
    try:
        print(f"env_id: {args.env_id}")
        print(f"observation_space: {env.observation_space}")
        print(f"action_space: {env.action_space}")

        obs, info = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        print(f"reset_obs_type: {type(obs).__name__}")
        print(f"reset_info_keys: {sorted(info.keys()) if isinstance(info, dict) else []}")
        print(f"step_obs_type: {type(next_obs).__name__}")
        print(f"reward: {reward}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")
        print(
            f"step_info_keys: {sorted(step_info.keys()) if isinstance(step_info, dict) else []}"
        )
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
