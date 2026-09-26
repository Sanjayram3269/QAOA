"""Paired statistical comparison of the frozen ML selector vs fixed C06.

The comparison uses exactly the same unseen test graph/noise observations for
both methods. No model is trained or tuned here and the test set is not used
to change the ML selector.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "data" / "ml"
MODEL_DIR = ML / "model_comparison"
SPLIT_DIR = ML / "splits"
OUT = ML / "final_analysis"
OUT.mkdir(parents=True, exist_ok=True)


def paired_test(df: pd.DataFrame, condition: str) -> dict:
    part = df if condition == "ALL" else df[df["noise_condition"] == condition]
    ml = part["ml_performance"].astype(float)
    fixed = part["fixed_c06_performance"].astype(float)
    diff = ml - fixed
    if len(part) > 1 and np.any(diff != 0):
        stat, p = wilcoxon(ml, fixed, alternative="two-sided", zero_method="wilcox")
    else:
        stat, p = np.nan, np.nan
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else np.nan
    mean_diff = float(diff.mean())
    se = sd / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
    return {
        "noise_condition": condition,
        "n": int(len(part)),
        "ml_mean": float(ml.mean()),
        "fixed_c06_mean": float(fixed.mean()),
        "mean_ml_minus_fixed": mean_diff,
        "median_ml_minus_fixed": float(diff.median()),
        "std_difference": sd,
        "ci95_low": float(mean_diff - 1.96 * se) if np.isfinite(se) else np.nan,
        "ci95_high": float(mean_diff + 1.96 * se) if np.isfinite(se) else np.nan,
        "wilcoxon_statistic": float(stat) if np.isfinite(stat) else np.nan,
        "wilcoxon_pvalue": float(p) if np.isfinite(p) else np.nan,
        "paired_cohens_d": float(mean_diff / sd) if np.isfinite(sd) and sd > 0 else np.nan,
        "ml_better_fraction": float((diff > 0).mean()),
    }


def main() -> None:
    print("=" * 72)
    print("NQComp 2027 — PAIRED ML vs FIXED C06 ANALYSIS")
    print("=" * 72)

    ml_perf_path = ML / "ml_performance.csv"
    test_graphs_path = SPLIT_DIR / "test_graphs.csv"
    selected_path = MODEL_DIR / "selected_test_results.csv"

    for path in (ml_perf_path, test_graphs_path, selected_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    ml_perf = pd.read_csv(ml_perf_path)
    test_graphs = pd.read_csv(test_graphs_path)["graph_id"].astype(str).tolist()
    selected = pd.read_csv(selected_path)

    required_selected = {"graph_id", "noise_condition", "selected_actual_approximation_ratio"}
    missing = required_selected - set(selected.columns)
    if missing:
        raise ValueError(f"Missing columns in selected_test_results.csv: {sorted(missing)}")

    # Fixed baseline is C06, selected from TRAIN in train_baselines.py.
    fixed = ml_perf[
        ml_perf["graph_id"].isin(test_graphs)
        & (ml_perf["config_id"] == "C06")
        & (ml_perf["noise_condition"].isin(["N0", "N1", "N2"]))
    ].copy()

    fixed = fixed[
        ["graph_id", "noise_condition", "mean_expected_approximation_ratio"]
    ].rename(columns={"mean_expected_approximation_ratio": "fixed_c06_performance"})

    ml = selected[
        ["graph_id", "noise_condition", "selected_actual_approximation_ratio"]
    ].rename(columns={"selected_actual_approximation_ratio": "ml_performance"})

    paired = ml.merge(
        fixed,
        on=["graph_id", "noise_condition"],
        how="inner",
        validate="one_to_one",
    )

    expected = len(test_graphs) * 3
    if len(paired) != expected:
        raise ValueError(
            f"Expected {expected} paired test observations ({len(test_graphs)} graphs × 3 noise conditions), "
            f"found {len(paired)}."
        )

    paired["ml_minus_fixed"] = paired["ml_performance"] - paired["fixed_c06_performance"]
    paired["relative_gain_percent"] = 100.0 * paired["ml_minus_fixed"] / paired["fixed_c06_performance"]
    paired = paired.sort_values(["noise_condition", "graph_id"]).reset_index(drop=True)
    paired.to_csv(OUT / "paired_ml_vs_fixed_results.csv", index=False)

    stats = pd.DataFrame([paired_test(paired, c) for c in ["ALL", "N0", "N1", "N2"]])
    stats.to_csv(OUT / "paired_ml_vs_fixed_statistics.csv", index=False)

    print(f"Test graphs: {len(test_graphs)}")
    print(f"Paired observations: {len(paired)}")
    print("\nPAIRED ML vs FIXED C06")
    print(stats.to_string(index=False))

    manifest = {
        "test_graphs": len(test_graphs),
        "paired_observations": len(paired),
        "fixed_configuration": "C06",
        "fixed_configuration_selected_from": "TRAIN",
        "model_selection_changed": False,
        "test_tuning_performed": False,
        "paired_test": "Wilcoxon signed-rank",
        "confidence_interval": "95% normal approximation for paired differences",
    }
    (OUT / "paired_ml_vs_fixed_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nOutput: {OUT}")
    print("\n✅ Paired ML-vs-fixed analysis completed without retraining or tuning.")


if __name__ == "__main__":
    main()
