"""Shared histogram entropy helper used across meshbench modules."""

from __future__ import annotations

import numpy as np


def _histogram_entropy(dots: np.ndarray, bins: int = 32) -> float:
    """Compute Shannon entropy of a dot-product distribution.

    Args:
        dots: 1-D array of values (typically in [-1, 1]).
        bins: Number of histogram bins.

    Returns:
        Entropy in bits. 0.0 if the array is empty or all in one bin.
    """
    if len(dots) == 0:
        return 0.0
    counts, _ = np.histogram(dots, bins=bins, range=(-1.0, 1.0))
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))
