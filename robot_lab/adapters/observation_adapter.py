from __future__ import annotations

from typing import Any


class ObservationAdapter:
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def adapt(self, obs: Any) -> Any:
        return obs


class IdentityObservationAdapter(ObservationAdapter):
    pass


class FlattenStateObservationAdapter(ObservationAdapter):
    def adapt(self, obs: Any) -> Any:
        if isinstance(obs, dict) and "state" in obs:
            return obs["state"]
        return obs
