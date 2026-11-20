"""Turn generated burst histories into model training examples."""

import numpy as np


def build_supervised_windows(workload: list, window_size: int = 3):
    """Return past-burst windows and the next burst that each window predicts."""
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    X, y = [], []
    for proc in workload:
        bursts = proc["bursts"]
        for t in range(window_size, len(bursts)):
            X.append(bursts[t - window_size : t])
            y.append(bursts[t])
    # Keep the feature matrix two-dimensional even when there are no examples.
    return np.asarray(X, dtype=float).reshape(-1, window_size), np.asarray(y, dtype=float)
