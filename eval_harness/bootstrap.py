from __future__ import annotations
"""
Bootstrap utilities for confidence intervals.

- Nonparametric bootstrap on arrays
- Works with any metric function that accepts numpy-like arrays
"""

from typing import Callable, Dict, Any, Sequence, Tuple
import numpy as np


def bootstrap_ci(
    metric_fn: Callable[..., float],
    args: Tuple[Sequence, ...],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute bootstrap mean and (1-alpha) CI for a metric function.

    Parameters
    ----------
    metric_fn : callable
        Function that returns a scalar metric, e.g., lambda yt, yp: f1(yt, yp)
    args : tuple of arrays
        Arguments passed to metric_fn (same length first dimension).
    n_boot : int
        Number of bootstrap resamples.
    alpha : float
        1 - confidence level (0.05 => 95% CI).
    seed : int
        RNG seed.

    Returns
    -------
    dict with keys: mean, lo, hi, n_boot
    """
    arrays = [np.asarray(a) for a in args]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("bootstrap_ci: all arrays must have the same length")

    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    indices = np.arange(n)

    for i in range(n_boot):
        b = rng.choice(indices, size=n, replace=True)
        boot_args = tuple(a[b] for a in arrays)
        samples[i] = float(metric_fn(*boot_args))

    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return {"mean": float(samples.mean()), "lo": lo, "hi": hi, "n_boot": int(n_boot)}
