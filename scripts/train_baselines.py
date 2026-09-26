"""Compute non-ML baselines on the frozen graph-level test split.

Baselines:
1. Random configuration: expected performance averaged over the test set.
2. Global best fixed configuration: configuration with the highest mean
   performance on TRAIN ONLY, evaluated on unseen TEST graphs.
3. Oracle: best measured configuration per TEST graph (upper reference only).

No test performance is used to choose the fixed baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ml"
SPLITS = DATA / "splits"
PERF = SPLITS / "ml_performance_common.csv"
OUT = DATA / "baselines"
SEED = 2027
TARGET = "mean_expected_approximation_ratio"


def summarize(df: pd.DataFrame, label: str) -> dict:
    return {
        "label": label,
        "mean": float(df[TARGET].mean()),
        "std": float(df[TARGET].std(ddof=1)),
        "median": float(df[TARGET].median()),
        "n": int(len(df)),
    }


def main() -> None:
    print("=" * 72)
    print("NQComp 2027 — NON-ML BASELINES")
    print("=" * 72)

    df = pd.read_csv(PERF)
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()

    # Fixed configuration is selected using TRAIN ONLY.
    train_scores = (
        train.groupby("config_id")[TARGET]
        .mean()
        .sort_values(ascending=False)
    )
    fixed_config = str(train_scores.index[0])

    # Evaluate fixed configuration on test, separately by noise condition.
    fixed = test[test["config_id"] == fixed_config].copy()

    # Oracle is computed only as an upper reference: best measured config per
    # graph and condition. It is NOT a deployable predictor.
    idx = test.groupby(["graph_id", "noise_condition"])[TARGET].idxmax()
    oracle = test.loc[idx].copy()

    # Random baseline: Monte Carlo over configuration choices per graph/condition.
    rng = np.random.default_rng(SEED)
    random_values = []
    keys = test[["graph_id", "noise_condition"]].drop_duplicates().sort_values(
        ["graph_id", "noise_condition"]
    )
    for _, key in keys.iterrows():
        candidates = test[
            (test["graph_id"] == key["graph_id"])
            & (test["noise_condition"] == key["noise_condition"])
        ]
        chosen = candidates.iloc[int(rng.integers(0, len(candidates)))]
        random_values.append(float(chosen[TARGET]))

    random_df = pd.DataFrame({TARGET: random_values})

    OUT.mkdir(parents=True, exist_ok=True)

    summaries = []
    for condition in ["ALL", "N0", "N1", "N2"]:
        if condition == "ALL":
            f = fixed
            o = oracle
            r = random_df
        else:
            f = fixed[fixed["noise_condition"] == condition]
            o = oracle[oracle["noise_condition"] == condition]
            r_keys = keys[keys["noise_condition"] == condition]
            # Reproduce the random samples corresponding to this condition.
            r = pd.DataFrame({TARGET: [
                float(test[(test["graph_id"] == row.graph_id) &
                           (test["noise_condition"] == row.noise_condition)]
                     .iloc[int(rng.integers(0, 12))][TARGET])
                for _, row in r_keys.iterrows()
            ]})
        for label, frame in [
            ("fixed", f),
            ("random", r),
            ("oracle", o),
        ]:
            s = summarize(frame, label)
            s["noise_condition"] = condition
            summaries.append(s)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT / "baseline_summary.csv", index=False)

    manifest = {
        "seed": SEED,
        "fixed_config_selected_from_train": fixed_config,
        "train_configuration_means": train_scores.to_dict(),
        "test_graphs": int(test["graph_id"].nunique()),
        "test_rows": int(len(test)),
        "oracle_is_reference_only": True,
        "random_is_seeded_monte_carlo": True,
    }
    (OUT / "baseline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Train graphs: {train['graph_id'].nunique()}")
    print(f"Test graphs : {test['graph_id'].nunique()}")
    print(f"Fixed config selected from TRAIN: {fixed_config}")
    print("\nTraining mean by configuration:")
    print(train_scores.to_string())
    print("\nBaseline summary:")
    print(summary_df.to_string(index=False))
    print(f"\nOutput: {OUT}")
    print("\n✅ Baselines computed without using test data to select the fixed configuration.")


if __name__ == "__main__":
    main()
