from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np

from robot_lab.policies.base import Policy
from robot_lab.runtime.types import ActionResult

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

_MISSING = object()


class OpenPIPolicy(Policy):
    """Adapter from this harness policy interface to OpenPI policies."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params=params)
        self.mode = str(self.params.get("mode", "local"))
        self.prompt = str(self.params.get("prompt", ""))
        self.action_key = str(self.params.get("action_key", "actions"))
        self.open_loop_horizon = int(self.params.get("open_loop_horizon", 1))
        if self.open_loop_horizon < 1:
            raise ValueError("OpenPIPolicy param 'open_loop_horizon' must be >= 1.")

        self.observation_cfg = dict(self.params.get("observation", {}))
        self.squeeze_batch_dim = bool(self.params.get("squeeze_batch_dim", True))
        self._client = self._build_client()
        self._action_queue: list[np.ndarray] = []
        self._last_infer_debug: dict[str, Any] = {}

    def reset(self, episode_context: dict) -> None:
        del episode_context
        self._action_queue.clear()
        self._last_infer_debug = {}
        reset = getattr(self._client, "reset", None)
        if callable(reset):
            reset()

    def act(
        self,
        obs: Any,
        info: dict,
        action_space: gym.Space,
        t: int,
    ) -> ActionResult:
        del info
        if not self._action_queue:
            openpi_obs = self._build_openpi_observation(obs)
            start = time.perf_counter()
            outputs = self._client.infer(openpi_obs)
            latency_ms = (time.perf_counter() - start) * 1000.0
            self._action_queue = self._extract_action_queue(outputs)
            self._last_infer_debug = {
                "mode": self.mode,
                "observation_keys": sorted(openpi_obs.keys()),
                "output_keys": sorted(str(k) for k in outputs.keys()),
                "queued_actions": len(self._action_queue),
                "policy_timing": outputs.get("policy_timing"),
            }
        else:
            latency_ms = None

        action = self._format_action_for_space(self._action_queue.pop(0), action_space)
        return ActionResult(
            action=action,
            command={"policy": "openpi", "mode": self.mode, "t": t},
            debug=self._last_infer_debug,
            latency_ms=latency_ms,
        )

    def _build_client(self) -> Any:
        if self.mode == "local":
            return self._build_local_policy()
        if self.mode in {"websocket", "remote"}:
            return self._build_websocket_policy()
        raise ValueError("OpenPIPolicy param 'mode' must be one of: local, websocket.")

    def _build_local_policy(self) -> Any:
        config_name = self.params.get("config_name")
        checkpoint_dir = self.params.get("checkpoint_dir")
        if not config_name or not checkpoint_dir:
            raise ValueError(
                "OpenPIPolicy local mode requires params 'config_name' and 'checkpoint_dir'."
            )

        try:
            from openpi.policies import policy_config
            from openpi.shared import download
            from openpi.training import config as openpi_config
        except ImportError as exc:  # pragma: no cover - depends on external OpenPI install
            raise ImportError(
                "OpenPIPolicy local mode requires OpenPI in this Python environment. "
                "Install Physical-Intelligence/openpi and its dependencies, then rerun."
            ) from exc

        config = openpi_config.get_config(str(config_name))
        resolved_checkpoint = download.maybe_download(str(checkpoint_dir))
        return policy_config.create_trained_policy(config, resolved_checkpoint)

    def _build_websocket_policy(self) -> Any:
        host = str(self.params.get("host", "localhost"))
        port = self.params.get("port", 8000)
        api_key = self.params.get("api_key")
        try:
            from openpi_client import websocket_client_policy
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "OpenPIPolicy websocket mode requires the openpi-client package."
            ) from exc

        return websocket_client_policy.WebsocketClientPolicy(
            host=host,
            port=None if port is None else int(port),
            api_key=None if api_key is None else str(api_key),
        )

    def _build_openpi_observation(self, obs: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}

        state_cfg = self.observation_cfg.get("state", _MISSING)
        has_default_state = (
            not isinstance(obs, Mapping)
            or (isinstance(obs, Mapping) and "state" in obs)
        )
        if state_cfg is not None and (state_cfg is not _MISSING or has_default_state):
            state_key = "observation/state"
            state_source = "state"
            if state_cfg is _MISSING:
                state_cfg = {}

            if isinstance(state_cfg, str):
                state_source = state_cfg
            elif isinstance(state_cfg, Mapping):
                state_key = str(state_cfg.get("key", state_key))
                state_source = state_cfg.get("source", state_source)

            if state_source == "state" and not isinstance(obs, Mapping):
                state = obs
            else:
                state = self._extract_path(obs, state_source)
            result[state_key] = self._to_numpy(state)

        for output_key, image_cfg in dict(self.observation_cfg.get("images", {})).items():
            source: Any = image_cfg
            resize = self.observation_cfg.get("image_resize")
            if isinstance(image_cfg, Mapping):
                source = image_cfg.get("source")
                resize = image_cfg.get("resize", resize)
            if source is None:
                raise ValueError(f"Image mapping for '{output_key}' is missing 'source'.")
            image = self._to_numpy(self._extract_path(obs, source))
            result[str(output_key)] = self._prepare_image(image, resize)

        extra = self.observation_cfg.get("extra", {})
        for output_key, source in dict(extra).items():
            result[str(output_key)] = self._to_numpy(self._extract_path(obs, source))

        prompt = self.observation_cfg.get("prompt", self.prompt)
        if prompt:
            result["prompt"] = str(prompt)

        return result

    def _extract_action_queue(self, outputs: Mapping[str, Any]) -> list[np.ndarray]:
        if self.action_key not in outputs:
            available = ", ".join(sorted(str(key) for key in outputs.keys()))
            raise KeyError(
                f"OpenPI output did not contain action key '{self.action_key}'. "
                f"Available keys: {available}"
            )

        actions = self._to_numpy(outputs[self.action_key])
        if actions.ndim == 0:
            raise ValueError("OpenPI action output must have at least one dimension.")
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim == 1:
            actions = actions[None, :]

        horizon = min(self.open_loop_horizon, int(actions.shape[0]))
        return [np.asarray(actions[i]) for i in range(horizon)]

    def _format_action_for_space(self, action: np.ndarray, action_space: gym.Space) -> Any:
        if not isinstance(action_space, gym.spaces.Box):
            return action

        action = np.asarray(action, dtype=action_space.dtype)
        if action.shape == action_space.shape:
            return action
        if (
            len(action_space.shape) == 2
            and action_space.shape[0] == 1
            and action.shape == action_space.shape[1:]
        ):
            return action[None, :]
        if action.size == int(np.prod(action_space.shape)):
            return action.reshape(action_space.shape)
        raise ValueError(
            "OpenPI action shape does not match ManiSkill action space: "
            f"action shape {action.shape}, action space shape {action_space.shape}."
        )

    def _prepare_image(self, image: np.ndarray, resize: Any) -> np.ndarray:
        image = self._squeeze_leading_batch(image)
        if resize is None:
            return self._to_uint8_image(image)

        if isinstance(resize, int):
            height = width = resize
        elif isinstance(resize, Sequence) and len(resize) == 2:
            height, width = int(resize[0]), int(resize[1])
        else:
            raise ValueError("Image resize must be an int or a [height, width] pair.")

        try:
            from openpi_client import image_tools
        except ImportError:
            return self._resize_with_pad(image, height, width)

        return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, height, width))

    def _resize_with_pad(self, image: np.ndarray, height: int, width: int) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Image resizing for OpenPI observations requires openpi-client or Pillow."
            ) from exc

        image = self._to_uint8_image(image)
        pil_image = Image.fromarray(image)
        scale = min(width / pil_image.width, height / pil_image.height)
        resized_size = (
            max(1, int(round(pil_image.width * scale))),
            max(1, int(round(pil_image.height * scale))),
        )
        pil_image = pil_image.resize(resized_size, Image.BILINEAR)

        canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
        offset = ((width - resized_size[0]) // 2, (height - resized_size[1]) // 2)
        canvas.paste(pil_image, offset)
        return np.asarray(canvas, dtype=np.uint8)

    def _to_uint8_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.dtype == np.uint8:
            return image
        if np.issubdtype(image.dtype, np.floating):
            max_value = float(np.nanmax(image)) if image.size else 1.0
            if max_value <= 1.0:
                image = image * 255.0
        return np.clip(image, 0, 255).astype(np.uint8)

    def _extract_path(self, value: Any, path: Any) -> Any:
        if path in (None, "", "."):
            return value
        current = value
        for part in str(path).replace(".", "/").split("/"):
            if not part:
                continue
            if isinstance(current, Mapping):
                if part not in current:
                    raise KeyError(f"Observation path '{path}' missing key '{part}'.")
                current = current[part]
            elif isinstance(current, (list, tuple)):
                current = current[int(part)]
            else:
                current = getattr(current, part)
        return current

    def _to_numpy(self, value: Any) -> np.ndarray:
        if torch is not None and torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if self.squeeze_batch_dim:
            array = self._squeeze_leading_batch(array)
        return array

    def _squeeze_leading_batch(self, array: np.ndarray) -> np.ndarray:
        if self.squeeze_batch_dim and array.ndim > 0 and array.shape[0] == 1:
            return array[0]
        return array
