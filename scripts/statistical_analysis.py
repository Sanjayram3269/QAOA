"""
NQComp 2027 — Statistical Analysis of Frozen ML Test Results

IMPORTANT:
- No model training.
- No hyperparameter tuning.
- No test-set model selection.
- Operates only on the already-frozen final test results.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "data" / "ml" / "model_comparison"
BASELINE_DIR = ROOT / "data" / "ml" / "baselines"
OUT = ROOT / "data" / "ml" / "final_analysis"

OUT.mkdir(parents=True, exist_ok=True)


def paired_stats(df: pd.DataFrame, condition: str) -> dict:
    """
    Paired comparison: ML-selected actual performance vs oracle
    on exactly the same graph/noise observations.
    """

    if condition == "ALL":
        d = df.copy()
    else:
        d = df[df["noise_condition"] == condition].copy()

    ml = d["selected_actual_approximation_ratio"].astype(float)
    oracle = d["oracle_approximation_ratio"].astype(float)

    difference = ml - oracle

    # Wilcoxon signed-rank test.
    # Zero differences are ignored by scipy's wilcoxon.
    try:
        stat, pvalue = wilcoxon(
            ml,
            oracle,
            alternative="two-sided",
            zero_method="wilcox",
        )
    except ValueError:
        stat = np.nan
        pvalue = np.nan

    n = len(d)

    mean_diff = float(difference.mean())
    median_diff = float(difference.median())

    if n > 1:
        std_diff = float(difference.std(ddof=1))
        se = std_diff / np.sqrt(n)
        ci_low = mean_diff - 1.96 * se
        ci_high = mean_diff + 1.96 * se

        # Paired Cohen's d.
        if std_diff > 0:
            cohens_d = mean_diff / std_diff
        else:
            cohens_d = np.nan
    else:
        std_diff = np.nan
        ci_low = np.nan
        ci_high = np.nan
        cohens_d = np.nan

    return {
        "noise_condition": condition,
        "n": n,
        "ml_mean": float(ml.mean()),
        "oracle_mean": float(oracle.mean()),
        "mean_ml_minus_oracle": mean_diff,
        "median_ml_minus_oracle": median_diff,
        "std_difference": std_diff,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "wilcoxon_statistic": float(stat) if not np.isnan(stat) else np.nan,
        "wilcoxon_pvalue": float(pvalue) if not np.isnan(pvalue) else np.nan,
        "paired_cohens_d": float(cohens_d) if not np.isnan(cohens_d) else np.nan,
    }


def main():

    print("=" * 72)
    print("NQComp 2027 — FROZEN TEST STATISTICAL ANALYSIS")
    print("=" * 72)

    results_file = MODEL_DIR / "selected_test_results.csv"

    if not results_file.exists():
        raise FileNotFoundError(
            f"Missing:\n{results_file}\n\n"
            "Run compare_ml_models.py first."
        )

    df = pd.read_csv(results_file)

    required = {
        "graph_id",
        "noise_condition",
        "config_id",
        "selected_actual_approximation_ratio",
        "oracle_approximation_ratio",
        "regret",
        "oracle_config_id",
        "selected_is_oracle",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    print(
        f"Test graph-condition observations: {len(df)}"
    )
    print(
        f"Test graphs: {df['graph_id'].nunique()}"
    )

    # ================================================================
    # 1. COMPLETE PER-GRAPH TABLE
    # ================================================================

    per_graph = df.copy()

    per_graph["ml_minus_oracle"] = (
        per_graph["selected_actual_approximation_ratio"]
        - per_graph["oracle_approximation_ratio"]
    )

    per_graph["relative_oracle_gap_percent"] = (
        100
        * per_graph["regret"]
        / per_graph["oracle_approximation_ratio"]
    )

    per_graph = per_graph.sort_values(
        ["noise_condition", "graph_id"]
    )

    per_graph.to_csv(
        OUT / "per_graph_test_results.csv",
        index=False,
    )

    # ================================================================
    # 2. STATISTICAL TESTS
    # ================================================================

    conditions = ["ALL", "N0", "N1", "N2"]

    stats = pd.DataFrame(
        [paired_stats(df, c) for c in conditions]
    )

    stats.to_csv(
        OUT / "paired_ml_vs_oracle_statistics.csv",
        index=False,
    )

    print("\nPAIRED ML vs ORACLE")
    print(stats.to_string(index=False))

    # ================================================================
    # 3. BASELINE SUMMARY
    # ================================================================

    baseline_file = BASELINE_DIR / "baseline_summary.csv"

    if baseline_file.exists():

        baseline = pd.read_csv(baseline_file)

        fixed = baseline[
            baseline["label"] == "fixed"
        ].copy()

        fixed = fixed[
            ["noise_condition", "mean", "std", "median", "n"]
        ].rename(
            columns={
                "mean": "fixed_mean",
                "std": "fixed_std",
                "median": "fixed_median",
                "n": "fixed_n",
            }
        )

        ml_summary = (
            df.groupby("noise_condition")
            ["selected_actual_approximation_ratio"]
            .agg(["mean", "std", "median", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "ml_mean",
                    "std": "ml_std",
                    "median": "ml_median",
                    "count": "ml_n",
                }
            )
        )

        comparison = ml_summary.merge(
            fixed,
            on="noise_condition",
            how="left",
        )

        comparison["ml_minus_fixed"] = (
            comparison["ml_mean"]
            - comparison["fixed_mean"]
        )

        comparison["relative_gain_percent"] = (
            100
            * comparison["ml_minus_fixed"]
            / comparison["fixed_mean"]
        )

        comparison.to_csv(
            OUT / "ml_vs_fixed_summary.csv",
            index=False,
        )

        print("\nML vs FIXED SUMMARY")
        print(comparison.to_string(index=False))

    # ================================================================
    # 4. GRAPH-LEVEL AGGREGATE
    # ================================================================

    graph_level = (
        df.groupby("graph_id")
        .agg(
            ml_mean=(
                "selected_actual_approximation_ratio",
                "mean",
            ),
            oracle_mean=(
                "oracle_approximation_ratio",
                "mean",
            ),
            regret_mean=("regret", "mean"),
        )
        .reset_index()
    )

    graph_level["ml_minus_oracle"] = (
        graph_level["ml_mean"]
        - graph_level["oracle_mean"]
    )

    graph_level.to_csv(
        OUT / "graph_level_summary.csv",
        index=False,
    )

    # ================================================================
    # 5. FIGURE — ML VS ORACLE BY GRAPH
    # ================================================================

    plt.figure(figsize=(12, 6))

    x = np.arange(len(graph_level))

    plt.plot(
        x,
        graph_level["ml_mean"],
        marker="o",
        label="ML selector",
    )

    plt.plot(
        x,
        graph_level["oracle_mean"],
        marker="o",
        label="Oracle",
    )

    plt.xlabel("Test graph index")
    plt.ylabel("Mean expected approximation ratio")
    plt.title("ML Selector vs Oracle Across Unseen Test Graphs")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUT / "ml_vs_oracle_test_graphs.png",
        dpi=300,
    )

    plt.close()

    # ================================================================
    # 6. FIGURE — NOISE CONDITIONS
    # ================================================================

    noise = (
        df.groupby("noise_condition")
        .agg(
            ml=(
                "selected_actual_approximation_ratio",
                "mean",
            ),
            oracle=(
                "oracle_approximation_ratio",
                "mean",
            ),
        )
        .reindex(["N0", "N1", "N2"])
    )

    plt.figure(figsize=(8, 6))

    x = np.arange(len(noise))
    width = 0.35

    plt.bar(
        x - width / 2,
        noise["ml"],
        width,
        label="ML selector",
    )

    plt.bar(
        x + width / 2,
        noise["oracle"],
        width,
        label="Oracle",
    )

    plt.xticks(
        x,
        ["N0", "N1", "N2"],
    )

    plt.xlabel("Noise condition")
    plt.ylabel("Mean expected approximation ratio")
    plt.title("ML Selector vs Oracle Under Noise")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUT / "ml_vs_oracle_noise_conditions.png",
        dpi=300,
    )

    plt.close()

    # ================================================================
    # 7. MANIFEST
    # ================================================================

    manifest = {
        "test_graphs": int(df["graph_id"].nunique()),
        "test_graph_condition_pairs": int(len(df)),
        "analysis_type": "post_selection_frozen_test_analysis",
        "model_retrained": False,
        "model_tuned": False,
        "test_used_for_model_selection": False,
        "paired_test": "Wilcoxon signed-rank",
        "confidence_interval": "95% normal approximation for paired differences",
        "outputs": [
            "per_graph_test_results.csv",
            "paired_ml_vs_oracle_statistics.csv",
            "ml_vs_fixed_summary.csv",
            "graph_level_summary.csv",
            "ml_vs_oracle_test_graphs.png",
            "ml_vs_oracle_noise_conditions.png",
        ],
    }

    with open(
        OUT / "statistical_analysis_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("OUTPUT")
    print("=" * 72)
    print(OUT)

    print(
        "\n✅ Statistical analysis completed."
    )
    print(
        "✅ No training, tuning, or test-set model selection performed."
    )


if __name__ == "__main__":
    main()