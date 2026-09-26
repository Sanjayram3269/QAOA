"""Train and evaluate a leakage-safe ML model for QAOA configuration selection.

The model predicts mean expected approximation ratio from graph/configuration/
noise features. Test graphs are never used for training or model selection.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ml" / "splits" / "ml_performance_common.csv"
OUT = ROOT / "data" / "ml" / "models"
TARGET = "mean_expected_approximation_ratio"

DROP = {"graph_id", "graph_seed", "split", TARGET, "optimizer_code", "noise_code", "family_code"}
CATEGORICAL = ["optimizer", "graph_family", "noise_condition", "config_id"]
NUMERIC = ["depth", "shots", "num_nodes", "num_edges", "density"]


def metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    print("=" * 72)
    print("NQComp 2027 — ML PERFORMANCE PREDICTOR")
    print("=" * 72)

    df = pd.read_csv(DATA)
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()

    features = NUMERIC + CATEGORICAL
    X_train, y_train = train[features], train[TARGET]
    X_val, y_val = val[features], val[TARGET]
    X_test, y_test = test[features], test[TARGET]

    preprocess = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=2027,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)

    val_pred = pipe.predict(X_val)
    test_pred = pipe.predict(X_test)

    print(f"Train rows: {len(train)} | graphs: {train.graph_id.nunique()}")
    print(f"Validation rows: {len(val)} | graphs: {val.graph_id.nunique()}")
    print(f"Test rows: {len(test)} | graphs: {test.graph_id.nunique()}")
    print("\nValidation metrics:", metrics(y_val, val_pred))
    print("Test metrics:", metrics(y_test, test_pred))

    pred = test[["graph_id", "config_id", "noise_condition", TARGET]].copy()
    pred["predicted_approximation_ratio"] = test_pred
    pred.to_csv(OUT / "test_predictions.csv", index=False)

    # Selector: choose the configuration with highest predicted performance
    # independently for every unseen graph and noise condition.
    idx = pred.groupby(["graph_id", "noise_condition"])["predicted_approximation_ratio"].idxmax()
    selected = pred.loc[idx].copy()
    actual = test[["graph_id", "config_id", "noise_condition", TARGET]].rename(
        columns={TARGET: "selected_actual_approximation_ratio"}
    )
    selected = selected.merge(actual, on=["graph_id", "config_id", "noise_condition"], how="left")

    # Oracle for reference only.
    oracle_idx = test.groupby(["graph_id", "noise_condition"])[TARGET].idxmax()
    oracle = test.loc[oracle_idx, ["graph_id", "noise_condition", TARGET]].rename(
        columns={TARGET: "oracle_approximation_ratio"}
    )
    selected = selected.merge(oracle, on=["graph_id", "noise_condition"], how="left")
    selected["regret"] = selected["oracle_approximation_ratio"] - selected["selected_actual_approximation_ratio"]
    selected.to_csv(OUT / "selector_test_results.csv", index=False)

    summary = []
    for condition in ["ALL", "N0", "N1", "N2"]:
        s = selected if condition == "ALL" else selected[selected.noise_condition == condition]
        summary.append({
            "noise_condition": condition,
            "graphs": int(s.graph_id.nunique()),
            "selected_mean": float(s.selected_actual_approximation_ratio.mean()),
            "oracle_mean": float(s.oracle_approximation_ratio.mean()),
            "mean_regret": float(s.regret.mean()),
            "zero_regret_fraction": float((s.regret.abs() < 1e-12).mean()),
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT / "selector_summary.csv", index=False)

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": "RandomForestRegressor",
        "n_estimators": 300,
        "random_state": 2027,
        "min_samples_leaf": 2,
        "features": features,
        "target": TARGET,
        "train_graphs": int(train.graph_id.nunique()),
        "validation_graphs": int(val.graph_id.nunique()),
        "test_graphs": int(test.graph_id.nunique()),
        "validation_metrics": metrics(y_val, val_pred),
        "test_metrics": metrics(y_test, test_pred),
    }
    (OUT / "ml_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nSelector summary:")
    print(summary_df.to_string(index=False))
    print(f"\nOutput: {OUT}")
    print("\n✅ ML predictor trained without test-graph training or tuning.")


if __name__ == "__main__":
    main()
