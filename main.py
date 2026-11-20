"""Run the experiment described in ``configs/experiment.yaml``."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from scipy.stats import wilcoxon

from src.data.generate_workload import clone_workload, generate_workload
from src.data.preprocess import build_supervised_windows
from src.evaluation.metrics import compute_run_metrics
from src.models.exponential_averaging import ExponentialAveragingPredictor
from src.models.linear_regression import LinearRegressionPredictor
from src.models.random_forest import RandomForestPredictor
from src.scheduling.simulator import SchedulingSimulator

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
RESULTS_PATH = ROOT / "results"


def load_config():
    """Read experiment settings from the YAML file."""
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def make_predictors(config, X_train, y_train, seed):
    """Create and train the three prediction methods for one trial."""
    model_settings = config["models"]
    window = model_settings["history_window"]
    default = model_settings["initial_burst_estimate"]

    # Exponential averaging uses the same interface but has no fit step.
    exponential = ExponentialAveragingPredictor(
        alpha=model_settings["exp_avg_alpha"], initial_estimate=default
    ).fit(X_train, y_train)

    linear = LinearRegressionPredictor(
        window_size=window, default_estimate=default
    ).fit(X_train, y_train)

    forest = RandomForestPredictor(
        window_size=window,
        n_estimators=model_settings["rf_n_estimators"],
        max_depth=model_settings["rf_max_depth"],
        seed=seed,
        default_estimate=default,
    ).fit(X_train, y_train)
    return exponential, linear, forest


def make_test_workload(config, noise, seed):
    """Build the held-out workload used to compare scheduling policies."""
    settings = config["experiment"]
    workload_settings = config["workload"]
    return generate_workload(
        n_processes=settings["n_test_processes"],
        bursts_per_process=settings["bursts_per_process"],
        noise_level=noise,
        seed=seed,
        arrival_scale=workload_settings["arrival_rate_scale"],
        base_mean=workload_settings["base_burst_mean"],
    )


def make_train_workload(config, noise, seed):
    """Make a separate workload so test bursts are not used for training."""
    settings = config["experiment"]
    workload_settings = config["workload"]
    return generate_workload(
        n_processes=settings["n_train_processes"],
        bursts_per_process=settings["bursts_per_process"],
        noise_level=noise,
        seed=seed,
        arrival_scale=workload_settings["arrival_rate_scale"],
        base_mean=workload_settings["base_burst_mean"],
    )


def run_experiments(config):
    """Train models, run every policy, and return one row per trial."""
    records = []
    experiment = config["experiment"]
    models = config["models"]
    scheduler = config["scheduler"]

    for regime, noise in config["workload"]["predictability_levels"].items():
        for trial in range(experiment["n_trials"]):
            seed = experiment["base_seed"] + trial

            # Adding a fixed offset gives training a different random stream.
            train_workload = make_train_workload(config, noise, seed + 10_000)
            test_workload = make_test_workload(config, noise, seed)
            X_train, y_train = build_supervised_windows(
                train_workload, window_size=models["history_window"]
            )
            predictors = make_predictors(config, X_train, y_train, seed)

            # Each policy receives its own copy because simulation updates state.
            policies = [
                ("FCFS", "fcfs", None),
                ("Round Robin", "rr", None),
                ("Ideal SJF (Oracle)", "ideal_sjf", None),
                ("Pred-SJF (ExpAvg)", "predicted_sjf", predictors[0]),
                ("Pred-SJF (LinearReg)", "predicted_sjf", predictors[1]),
                ("Pred-SJF (RandomForest)", "predicted_sjf", predictors[2]),
            ]

            for label, policy, predictor in policies:
                simulator = SchedulingSimulator(
                    policy=policy,
                    predictor=predictor,
                    time_quantum=scheduler["time_quantum"],
                    cs_cost=scheduler["context_switch_cost"],
                    io_delay=scheduler["io_delay"],
                    charge_overhead=True,
                )
                result = simulator.run(clone_workload(test_workload))
                records.append({
                    "regime": regime,
                    "trial": trial,
                    "method": label,
                    **compute_run_metrics(result),
                })

    return pd.DataFrame(records)


def save_summary(results):
    """Save average and standard deviation for the manuscript results table."""
    measures = {
        "avg_waiting_ms": "Avg Wait (ms)",
        "p95_waiting_ms": "P95 Wait (ms)",
        "avg_turnaround_ms": "Avg Turnaround (ms)",
        "mae_ms": "MAE (ms)",
    }
    rows = []
    for (regime, method), group in results.groupby(["regime", "method"], sort=False):
        row = {"Regime": regime, "Method": method}
        for column, label in measures.items():
            # Prediction error is not applicable to non-predictive policies.
            if column == "mae_ms" and not method.startswith("Pred-SJF"):
                row[label] = "-"
            else:
                row[label] = f"{group[column].mean():.2f} +/- {group[column].std():.2f}"
        row["Inversion Rate"] = f"{group['inversion_rate'].mean():.3f}"
        row["Prediction Wall Time (ms)"] = f"{group['pred_overhead_ms'].mean():.3f}"
        row["Prediction CPU Time (ms)"] = f"{group['pred_cpu_ms'].mean():.3f}"
        row["Peak Python Memory (KB)"] = f"{group['peak_memory_kb'].mean():.1f}"
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_PATH / "tables" / "summary_results.csv", index=False)
    return summary


def compare_with_round_robin(results, regimes):
    """Print paired Wilcoxon results for predicted SJF and Round Robin."""
    print("\nWilcoxon signed-rank tests: average waiting time vs Round Robin")
    predicted_methods = [
        "Pred-SJF (ExpAvg)",
        "Pred-SJF (LinearReg)",
        "Pred-SJF (RandomForest)",
    ]
    for regime in regimes:
        group = results[results["regime"] == regime]
        rr = group[group["method"] == "Round Robin"].sort_values("trial")
        for method in predicted_methods:
            predicted = group[group["method"] == method].sort_values("trial")
            if len(rr) != len(predicted) or len(rr) == 0:
                print(f"{regime} | {method}: not enough matched trial results")
                continue
            if (rr["avg_waiting_ms"].to_numpy() == predicted["avg_waiting_ms"].to_numpy()).all():
                print(f"{regime} | {method}: no difference between paired trials")
                continue
            _, p_value = wilcoxon(
                rr["avg_waiting_ms"].to_numpy(),
                predicted["avg_waiting_ms"].to_numpy(),
            )
            print(f"{regime} | {method}: p = {p_value:.5f}")


def save_waiting_time_plot(results):
    """Draw the mean waiting-time comparison across predictability levels."""
    order = ["high_predictability", "medium_predictability", "low_predictability"]
    means = results.pivot_table(
        index="regime", columns="method", values="avg_waiting_ms", aggfunc="mean"
    ).reindex(order)
    axis = means.plot(kind="bar", figsize=(11, 5), rot=0)
    axis.set_ylabel("Average waiting time (ms)")
    axis.set_xlabel("Workload predictability")
    axis.set_title("Scheduling results with prediction overhead included")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "figures" / "waiting_time_by_predictability.png", dpi=300)
    plt.close()


def main():
    """Run the experiment and write its tables and figure to ``results``."""
    config = load_config()
    (RESULTS_PATH / "tables").mkdir(parents=True, exist_ok=True)
    (RESULTS_PATH / "figures").mkdir(parents=True, exist_ok=True)

    results = run_experiments(config)
    results.to_csv(RESULTS_PATH / "tables" / "raw_trial_metrics.csv", index=False)
    summary = save_summary(results)
    save_waiting_time_plot(results)
    print(summary.to_string(index=False))
    compare_with_round_robin(
        results, config["workload"]["predictability_levels"].keys()
    )
    print("\nSaved experiment tables and plot in the results/ folder.")


if __name__ == "__main__":
    main()
