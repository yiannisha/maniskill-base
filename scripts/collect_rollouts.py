#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_lab.adapters import build_action_adapter, build_observation_adapter
from robot_lab.policies import build_policy
from robot_lab.runtime import EpisodeEvaluator, EpisodeRecorder, RolloutRunner, make_env
from robot_lab.utils import ensure_dir, load_config, save_config, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect rollout data in ManiSkill.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config.")
    parser.add_argument(
        "--force-recording",
        action="store_true",
        help="Enable recording even if the config has recording.enabled=false.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    recording_cfg = cfg.setdefault("recording", {})

    if not recording_cfg.get("enabled", False):
        if not args.force_recording:
            raise SystemExit(
                "Recording is disabled in the config. Set recording.enabled=true or use --force-recording."
            )
        recording_cfg["enabled"] = True
        if not recording_cfg.get("save_video", False) and not recording_cfg.get(
            "save_trajectory", False
        ):
            recording_cfg["save_trajectory"] = True
        recording_cfg.setdefault("video_fps", 30)

    output_dir = ensure_dir(cfg["experiment"]["output_dir"])
    save_config(cfg, output_dir / "config.yaml")
    set_global_seed(cfg["experiment"].get("seed"))

    env = make_env(cfg)
    try:
        policy = build_policy(cfg.get("policy", {}))
        action_adapter = build_action_adapter(cfg.get("action_adapter"))
        observation_adapter = build_observation_adapter(cfg.get("observation_adapter"))
        evaluator = EpisodeEvaluator()
        recorder = EpisodeRecorder(output_dir)
        runner = RolloutRunner(
            cfg=cfg,
            env=env,
            policy=policy,
            evaluator=evaluator,
            recorder=recorder,
            action_adapter=action_adapter,
            observation_adapter=observation_adapter,
        )
        summary = runner.run()
    finally:
        env.close()

    print(json.dumps(summary, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
