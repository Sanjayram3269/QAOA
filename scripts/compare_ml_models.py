"""Compare ML regressors with graph-level leakage-safe validation.

Protocol:
- 62 train graphs are used for fitting candidate models.
- 13 validation graphs select the model using RMSE.
- The selected model is refit on train + validation.
- 14 test graphs are evaluated once for final selection performance.
- Test graphs are never used for model selection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ml" / "splits" / "ml_performance_common.csv"
OUT = ROOT / "data" / "ml" / "model_comparison"
TARGET = "mean_expected_approximation_ratio"
SEED = 2027
CATEGORICAL = ["optimizer", "graph_family", "noise_condition", "config_id"]
NUMERIC = ["depth", "shots", "num_nodes", "num_edges", "density"]
FEATURES = NUMERIC + CATEGORICAL


def metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def make_preprocessor():
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )


def candidates():
    return {
        "ridge": Pipeline([
            ("preprocess", make_preprocessor()),
            ("scale", StandardScaler(with_mean=False)),
            ("model", Ridge(alpha=1.0)),
        ]),
        "random_forest": Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=300, random_state=SEED, n_jobs=-1, min_samples_leaf=2
            )),
        ]),
        "extra_trees": Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", ExtraTreesRegressor(
                n_estimators=300, random_state=SEED, n_jobs=-1, min_samples_leaf=2
            )),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", HistGradientBoostingRegressor(
                max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
                l2_regularization=1.0, random_state=SEED
            )),
        ]),
    }


def selector_results(model, test):
    pred = test[["graph_id", "config_id", "noise_condition", TARGET]].copy()
    pred["predicted_approximation_ratio"] = model.predict(test[FEATURES])
    idx = pred.groupby(["graph_id", "noise_condition"])["predicted_approximation_ratio"].idxmax()
    selected = pred.loc[idx].copy()
    actual = test[["graph_id", "config_id", "noise_condition", TARGET]].rename(
        columns={TARGET: "selected_actual_approximation_ratio"}
    )
    selected = selected.merge(
        actual, on=["graph_id", "config_id", "noise_condition"], how="left"
    )
    oracle_idx = test.groupby(["graph_id", "noise_condition"])[TARGET].idxmax()
    oracle = test.loc[oracle_idx, ["graph_id", "noise_condition", TARGET]].rename(
        columns={TARGET: "oracle_approximation_ratio"}
    )
    selected = selected.merge(oracle, on=["graph_id", "noise_condition"], how="left")
    selected["regret"] = (
        selected["oracle_approximation_ratio"]
        - selected["selected_actual_approximation_ratio"]
    )
    selected["oracle_config_id"] = test.loc[oracle_idx, ["graph_id", "noise_condition", "config_id"]].set_index(
        ["graph_id", "noise_condition"]
    ).reindex(pd.MultiIndex.from_frame(selected[["graph_id", "noise_condition"]])).values
    selected["selected_is_oracle"] = selected["config_id"].eq(selected["oracle_config_id"])
    return selected


def summarize_selection(selected, model_name):
    rows = []
    for condition in ["ALL", "N0", "N1", "N2"]:
        s = selected if condition == "ALL" else selected[selected.noise_condition == condition]
        rows.append({
            "model": model_name,
            "noise_condition": condition,
            "graphs": int(s.graph_id.nunique()),
            "selected_mean": float(s.selected_actual_approximation_ratio.mean()),
            "oracle_mean": float(s.oracle_approximation_ratio.mean()),
            "mean_regret": float(s.regret.mean()),
            "oracle_selection_accuracy": float(s.selected_is_oracle.mean()),
        })
    return rows


def main():
    print("=" * 72)
    print("NQComp 2027 — VALIDATION-DRIVEN ML MODEL COMPARISON")
    print("=" * 72)
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)
    train = df[df.split == "train"].copy()
    val = df[df.split == "validation"].copy()
    test = df[df.split == "test"].copy()

    print(f"Train: {train.graph_id.nunique()} graphs / {len(train)} rows")
    print(f"Validation: {val.graph_id.nunique()} graphs / {len(val)} rows")
    print(f"Test: {test.graph_id.nunique()} graphs / {len(test)} rows")

    validation_rows = []
    fitted = {}
    for name, model in candidates().items():
        model.fit(train[FEATURES], train[TARGET])
        pred = model.predict(val[FEATURES])
        m = metrics(val[TARGET], pred)
        m["model"] = name
        validation_rows.append(m)
        fitted[name] = model
        print(f"\n{name}: validation MAE={m['mae']:.6f}, RMSE={m['rmse']:.6f}, R2={m['r2']:.6f}")

    validation_df = pd.DataFrame(validation_rows).sort_values("rmse")
    validation_df.to_csv(OUT / "validation_model_comparison.csv", index=False)
    best_name = str(validation_df.iloc[0]["model"])
    print(f"\nSelected using VALIDATION RMSE: {best_name}")

    # Freeze the model choice, then refit on train + validation.
    train_val = pd.concat([train, val], ignore_index=True)
    final_model = candidates()[best_name]
    final_model.fit(train_val[FEATURES], train_val[TARGET])
    test_pred = final_model.predict(test[FEATURES])
    test_metrics = metrics(test[TARGET], test_pred)
    print("\nFinal test prediction metrics:", test_metrics)

    selected = selector_results(final_model, test)
    selected.to_csv(OUT / "selected_test_results.csv", index=False)
    summary_df = pd.DataFrame(summarize_selection(selected, best_name))
    summary_df.to_csv(OUT / "selected_test_summary.csv", index=False)

    manifest = {
        "seed": SEED,
        "features": FEATURES,
        "target": TARGET,
        "selection_metric": "validation_rmse",
        "selected_model": best_name,
        "train_graphs": int(train.graph_id.nunique()),
        "validation_graphs": int(val.graph_id.nunique()),
        "test_graphs": int(test.graph_id.nunique()),
        "final_test_metrics": test_metrics,
    }
    (OUT / "model_comparison_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nFINAL TEST SELECTION SUMMARY")
    print(summary_df.to_string(index=False))
    print(f"\nOutput: {OUT}")
    print("\n✅ Model selection used validation only; final test evaluation was performed after freezing the model choice.")


if __name__ == "__main__":
    main()
