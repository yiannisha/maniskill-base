from __future__ import annotations

import random

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
