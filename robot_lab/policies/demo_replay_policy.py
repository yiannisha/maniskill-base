from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
from mani_skill import DEMO_DIR

from robot_lab.policies.base import Policy
from robot_lab.runtime.types import ActionResult


class DemoReplayPolicy(Policy):
    """Replay recorded ManiSkill demonstration actions episode-by-episode."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params=params)
        # `env_id` is kept as a backward-compatible alias for the demo dataset id.
        self.demo_env_id = str(
            self.params.get("demo_env_id", self.params.get("env_id", "PickCube-v1"))
        )
        self.demo_source = str(self.params.get("source", "motionplanning"))
        self.demo_control_mode = self.params.get("control_mode")
        self.cycle_episodes = bool(self.params.get("cycle_episodes", False))
        self.strict_env_id_match = bool(self.params.get("strict_env_id_match", False))
        self.trajectory_path = self._resolve_trajectory_path()
        self.metadata_path = self.trajectory_path.with_suffix(".json")
        self._load_metadata()
        self._h5 = h5py.File(self.trajectory_path, "r")
        self._current_demo_index: int | None = None
        self._current_demo_episode: dict[str, Any] | None = None
        self._current_actions: np.ndarray | None = None
        self._action_cursor = 0

    def _resolve_trajectory_path(self) -> Path:
        if "trajectory_path" in self.params:
            return Path(self.params["trajectory_path"]).expanduser().resolve()

        demo_root = Path(self.params.get("demo_root", DEMO_DIR)).expanduser().resolve()
        base_dir = demo_root / self.demo_env_id / self.demo_source
        if self.demo_source == "rl":
            control_mode = str(
                self.demo_control_mode or self.params.get("control_mode", "pd_joint_delta_pos")
            )
            filename = f"trajectory.none.{control_mode}.physx_cuda.h5"
        else:
            filename = "trajectory.h5"
        return base_dir / filename

    def _load_metadata(self) -> None:
        if not self.trajectory_path.is_file():
            raise FileNotFoundError(f"Demo trajectory file not found: {self.trajectory_path}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Demo metadata file not found: {self.metadata_path}")

        with self.metadata_path.open("r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        self._episodes = list(self._metadata.get("episodes", []))
        self._env_info = dict(self._metadata.get("env_info", {}))
        if not self._episodes:
            raise ValueError(f"No episodes found in demo metadata: {self.metadata_path}")

    def _select_demo_episode(self, episode_index: int) -> tuple[int, dict[str, Any]]:
        if self.cycle_episodes:
            demo_index = episode_index % len(self._episodes)
        else:
            if episode_index >= len(self._episodes):
                raise IndexError(
                    f"Requested evaluation episode {episode_index}, but only "
                    f"{len(self._episodes)} demos are available in {self.trajectory_path}."
                )
            demo_index = episode_index
        return demo_index, self._episodes[demo_index]

    def get_reset_kwargs(self, episode_context: dict[str, Any]) -> dict[str, Any]:
        runtime_env_id = episode_context.get("env_id")
        dataset_env_id = self._env_info.get("env_id")
        if (
            self.strict_env_id_match
            and runtime_env_id is not None
            and dataset_env_id is not None
            and runtime_env_id != dataset_env_id
        ):
            raise ValueError(
                f"Demo dataset env_id {dataset_env_id} does not match requested env_id "
                f"{runtime_env_id}."
            )
        control_mode = episode_context.get("control_mode")

        demo_index, demo_episode = self._select_demo_episode(
            int(episode_context["episode_index"])
        )
        demo_control_mode = demo_episode.get("control_mode")
        if (
            control_mode is not None
            and demo_control_mode is not None
            and control_mode != demo_control_mode
        ):
            raise ValueError(
                f"Demo control_mode {demo_control_mode} does not match "
                f"requested control_mode {control_mode}."
            )
        trajectory_key = f"traj_{demo_episode['episode_id']}"
        self._current_actions = self._h5[trajectory_key]["actions"][:]
        self._current_demo_index = demo_index
        self._current_demo_episode = demo_episode
        self._action_cursor = 0

        reset_kwargs = dict(demo_episode.get("reset_kwargs", {}))
        if "seed" not in reset_kwargs and "episode_seed" in demo_episode:
            reset_kwargs["seed"] = demo_episode["episode_seed"]
        return reset_kwargs

    def reset(self, episode_context: dict) -> None:
        del episode_context
        self._action_cursor = 0

    def act(
        self,
        obs: Any,
        info: dict,
        action_space: gym.Space,
        t: int,
    ) -> ActionResult:
        del obs, info, action_space, t
        if self._current_actions is None or self._current_demo_episode is None:
            raise RuntimeError("No demo episode is loaded. Call get_reset_kwargs() first.")

        if self._action_cursor >= len(self._current_actions):
            raise IndexError(
                f"Demo actions exhausted for episode_id "
                f"{self._current_demo_episode['episode_id']}."
            )

        action = np.asarray(self._current_actions[self._action_cursor], dtype=np.float32)
        self._action_cursor += 1
        return ActionResult(
            action=action,
            debug={
                "demo_source": self.demo_source,
                "demo_env_id": self.demo_env_id,
                "demo_index": self._current_demo_index,
                "demo_episode_id": self._current_demo_episode["episode_id"],
                "demo_action_index": self._action_cursor - 1,
            },
        )
