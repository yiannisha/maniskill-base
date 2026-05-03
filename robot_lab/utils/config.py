from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from robot_lab.utils.io import ensure_dir


def _to_plain_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {config_path} must load to a dictionary.")
    return _to_plain_data(data)


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_to_plain_data(cfg), f, sort_keys=False)
