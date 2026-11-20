"""Calculate the results reported for one simulator run."""

import numpy as np


def compute_run_metrics(sim_output: dict) -> dict:
    completed = sim_output["completed"]
    all_waits = [w for p in completed for w in p.wait_times]
    all_turnarounds = [t for p in completed for t in p.turnaround_times]
    all_preds = [pr for p in completed for pr in p.predictions]
    all_actuals = [ac for p in completed for ac in p.actuals]

    # MAE is meaningful only for policies that made burst predictions.
    mae = (
        float(np.mean(np.abs(np.asarray(all_preds) - np.asarray(all_actuals))))
        if all_preds
        else 0.0
    )

    # Empty workloads have no wait or turnaround observations.
    avg_wait = float(np.mean(all_waits)) if all_waits else 0.0
    p95_wait = float(np.percentile(all_waits, 95)) if all_waits else 0.0
    avg_turnaround = float(np.mean(all_turnarounds)) if all_turnarounds else 0.0

    return {
        "avg_waiting_ms": avg_wait,
        "p95_waiting_ms": p95_wait,
        "avg_turnaround_ms": avg_turnaround,
        "mae_ms": mae,
        "inversion_rate": sim_output["inversion_rate"],
        "context_switches": sim_output["context_switches"],
        "pred_overhead_ms": sim_output["total_pred_overhead_ms"],
        "pred_cpu_ms": sim_output["total_pred_cpu_ms"],
        "peak_memory_kb": sim_output["peak_memory_kb"],
    }
