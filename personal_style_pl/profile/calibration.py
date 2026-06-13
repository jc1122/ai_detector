"""Temperature calibration + profile finalization (full impl in a later task)."""

from __future__ import annotations

import numpy as np


def finalize_profile(profile, matrix: np.ndarray) -> None:
    profile.temperature = 1.0
    profile.training_scores = []
