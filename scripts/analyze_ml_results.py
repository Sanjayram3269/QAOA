"""Paper-oriented statistical analysis of the frozen ML test results.

No model is trained or tuned here. The script only analyzes the already-frozen
14-graph test results and the baseline summary. Confidence intervals are
normal-approximation intervals over graph-level observations; with n=14 they
should be reported as uncertainty estimates, not as large-sample guarantees.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "ml" / "model_comparison"
BASE_DIR = ROOT / "data" / "ml" / "baselines"
OUT = ROOT / "data" / "ml" / "analysis"
SEED = 2027
Z = 1.96


def ci95(values: pd.Series) -> tuple[float, float]:
    x = values.dropna().astype(float).to_numpy()
    mean = float(np.mean(x))
    if len(x) <= 1:
        return mean, mean
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    return mean - Z * se, mean + Z * se


def aggregate(df: pd.DataFrame, condition: str) -> dict:
    s = df if condition == "ALL" else df[df["noise_condition"] == condition]
    selected = s["selected_actual_approximation_ratio"]
    oracle = s["oracle_approximation_ratio"]
    regret = s["regret"]
    low, high = ci95(selected)
    return {
        "noise_condition": condition,
        "n_graph_condition_pairs": int(len(s)),
        "graphs": int(s["graph_id"].nunique()),
        "selected_mean": float(selected.mean()),
        "selected_std": float(selected.std(ddof=1)),
        "selected_ci95_low": low,
        "selected_ci95_high": high,
        "oracle_mean": float(oracle.mean()),
        "mean_regret": float(regret.mean()),
        "regret_std": float(regret.std(ddof=1)),
        "oracle_selection_accuracy": float(s["selected_is_oracle"].mean()),
    }


def main() -> None:
    print("=" * 72)
    print("NQComp 2027 — ML RESULT ROBUSTNESS ANALYSIS")
    print("=" * 72)
    OUT.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(MODEL_DIR / "selected_test_results.csv")
    baselines = pd.read_csv(BASE_DIR / "baseline_summary.csv")

    required = {
        "graph_id", "noise_condition", "selected_actual_approximation_ratio",
        "oracle_approximation_ratio", "regret", "selected_is_oracle"
    }
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Add graph-level comparison rows against the fixed baseline. The fixed
    # baseline is represented in baseline_summary by noise condition, so the
    # comparison is made against the corresponding fixed mean. This is a
    # summary-level comparison, not a paired per-graph fixed run.
    fixed_by_condition = baselines[baselines["label"] == "fixed"].set_index("noise_condition")["mean"]
    selected["fixed_baseline_mean"] = selected["noise_condition"].map(fixed_by_condition)
    selected["gain_over_fixed_mean"] = (
        selected["selected_actual_approximation_ratio"] - selected["fixed_baseline_mean"]
    )

    selected.to_csv(OUT / "per_graph_results.csv", index=False)

    summary_rows = [aggregate(selected, c) for c in ["ALL", "N0", "N1", "N2"]]
    summary = pd.DataFrame(summary_rows)
    summary["fixed_baseline_mean"] = summary["noise_condition"].map(fixed_by_condition)
    summary["absolute_gain_over_fixed"] = summary["selected_mean"] - summary["fixed_baseline_mean"]
    summary["relative_gain_percent"] = (
        100 * summary["absolute_gain_over_fixed"] / summary["fixed_baseline_mean"]
    )
    summary.to_csv(OUT / "paper_summary.csv", index=False)

    noise_summary = summary[summary["noise_condition"] != "ALL"].copy()
    noise_summary.to_csv(OUT / "noise_summary.csv", index=False)

    ci = summary[[
        "noise_condition", "graphs", "selected_mean", "selected_ci95_low",
        "selected_ci95_high", "oracle_mean", "mean_regret"
    ]].copy()
    ci.to_csv(OUT / "confidence_intervals.csv", index=False)

    manifest = {
        "test_graphs": int(selected["graph_id"].nunique()),
        "test_graph_condition_pairs": int(len(selected)),
        "confidence_level": 0.95,
        "ci_method": "normal_approximation_over_graph_condition_observations",
        "z_value": Z,
        "model_selection_frozen_before_analysis": True,
        "no_training_or_tuning_performed": True,
        "seed": SEED,
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nPAPER SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nOutput: {OUT}")
    print("\n✅ Analysis completed without retraining or modifying experimental data.")


if __name__ == "__main__":
    main()
