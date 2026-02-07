"""Simple multipath bias model."""

from __future__ import annotations

import numpy as np


def multipath_bias_m(
    elev_deg: float,
    rng: np.random.Generator | None = None,
    max_bias_m: float = 1.5,
) -> float:
    """Return an elevation-dependent multipath bias."""

    elev_deg = float(elev_deg)
    elev_deg = np.clip(elev_deg, 0.0, 90.0)
    scale = np.exp(-elev_deg / 15.0)
    bias = max_bias_m * scale
    if rng is None:
        return float(bias)
    jitter = rng.normal(0.0, 0.2 * max_bias_m * scale)
    return float(bias + jitter)
