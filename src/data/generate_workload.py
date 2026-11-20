"""Create repeatable synthetic process arrivals and CPU burst sequences."""

import copy

import numpy as np


def generate_workload(n_processes: int, bursts_per_process: int, noise_level: float, seed: int, arrival_scale: float = 4.0, base_mean: float = 12.0):
    if n_processes < 0 or bursts_per_process < 1:
        raise ValueError("process count must be non-negative and each process needs bursts")
    if noise_level < 0 or arrival_scale <= 0 or base_mean <= 0:
        raise ValueError("noise must be non-negative; arrival_scale and base_mean must be positive")

    # A local random generator makes this workload independent of other code.
    rng = np.random.default_rng(seed)
    inter_arrivals = rng.exponential(scale=arrival_scale, size=n_processes)
    arrival_times = np.cumsum(inter_arrivals)

    workload = []
    # Higher noise means less similarity between consecutive bursts.
    phi = max(0.1, 0.85 - noise_level)

    for pid in range(n_processes):
        # Gamma variation gives different processes different average burst sizes.
        proc_mean = rng.gamma(shape=2.0, scale=base_mean / 2.0)
        bursts = []
        current = proc_mean
        for _ in range(bursts_per_process):
            shock = rng.normal(0.0, noise_level * proc_mean)
            current = phi * current + (1.0 - phi) * proc_mean + shock
            bursts.append(float(np.clip(current, 1.0, 150.0)))

        workload.append({
            "pid": pid,
            "arrival_time": float(arrival_times[pid]),
            "bursts": bursts,
        })
    return workload

def clone_workload(workload):
    return copy.deepcopy(workload)
