from __future__ import annotations

from typing import Any

from robot_lab.adapters.action_adapter import ClipBoxActionAdapter, IdentityActionAdapter
from robot_lab.adapters.observation_adapter import (
    FlattenStateObservationAdapter,
    IdentityObservationAdapter,
)

ACTION_ADAPTER_REGISTRY: dict[str, type] = {
    "identity": IdentityActionAdapter,
    "clip_box": ClipBoxActionAdapter,
}

OBSERVATION_ADAPTER_REGISTRY: dict[str, type] = {
    "identity": IdentityObservationAdapter,
    "flatten_state": FlattenStateObservationAdapter,
}


def build_action_adapter(adapter_cfg: dict[str, Any] | None) -> Any:
    cfg = adapter_cfg or {"type": "identity"}
    adapter_type = cfg.get("type", "identity")
    params = cfg.get("params", {})
    try:
        adapter_cls = ACTION_ADAPTER_REGISTRY[adapter_type]
    except KeyError as exc:
        available = ", ".join(sorted(ACTION_ADAPTER_REGISTRY))
        raise KeyError(f"Unknown action adapter '{adapter_type}'. Available: {available}") from exc
    return adapter_cls(params=params)


def build_observation_adapter(adapter_cfg: dict[str, Any] | None) -> Any:
    cfg = adapter_cfg or {"type": "identity"}
    adapter_type = cfg.get("type", "identity")
    params = cfg.get("params", {})
    try:
        adapter_cls = OBSERVATION_ADAPTER_REGISTRY[adapter_type]
    except KeyError as exc:
        available = ", ".join(sorted(OBSERVATION_ADAPTER_REGISTRY))
        raise KeyError(
            f"Unknown observation adapter '{adapter_type}'. Available: {available}"
        ) from exc
    return adapter_cls(params=params)


__all__ = [
    "ACTION_ADAPTER_REGISTRY",
    "OBSERVATION_ADAPTER_REGISTRY",
    "ClipBoxActionAdapter",
    "FlattenStateObservationAdapter",
    "IdentityActionAdapter",
    "IdentityObservationAdapter",
    "build_action_adapter",
    "build_observation_adapter",
]
