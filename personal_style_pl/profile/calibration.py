"""Temperature calibration from training self-distances."""

from __future__ import annotations

import math

import numpy as np

from ..config import Z_CLIP


def chunk_distance(profile, feature_row: np.ndarray) -> float:
    """Median robust z-distance over stable features (clipped). Returns 0.0 when the
    profile has no stable features (degenerate uniform corpus); real varied corpora
    always have stable features."""
    mask = profile.stable_mask
    center = profile.robust_center[mask]
    scale = profile.robust_scale[mask]
    z = np.abs((feature_row[mask] - center) / scale)
    z = np.clip(z, 0.0, Z_CLIP)
    return float(np.median(z)) if z.size else 0.0


def finalize_profile(profile, matrix: np.ndarray) -> None:
    distances = [chunk_distance(profile, row) for row in matrix]
    profile.training_scores = distances
    if len(distances) >= 5:
        # Calibrate temperature so a typical own-chunk (median self-distance) maps to a
        # z-component of ~80 (the close_to_my_style boundary) before the cosine blend:
        # 100*exp(-median/temperature) = 80  ->  temperature = median / -ln(0.80).
        median_d = float(np.median(distances))
        profile.temperature = max(median_d / -math.log(0.80), 0.5)
    else:
        profile.temperature = 1.0
