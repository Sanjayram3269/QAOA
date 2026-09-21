"""Prepare, train, and evaluate the leakage-safe QAOA configuration selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib

# Make `python scripts/run_ml_pipeline.py` work from a fresh repository clone.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.features import build_features_from_raw_results, build_ml_features
from src.ml_dataset import build_ml_dataset, save_ml_dataset
from src.ml_selector import run_selector_experiment
from src.splitting import attach_graph_splits, split_graphs
from src.validation import validate_raw_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Validated raw QAOA CSV")
    parser.add_argument(
        "--output-dir",
        default="data/processed/ml_selector",
        help="Directory for prepared data, predictions, metrics, and model",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = validate_raw_csv(args.input)
    aggregated, labels = build_ml_dataset(raw)
    graph_features = build_features_from_raw_results(raw)
    dataset = build_ml_features(aggregated, labels, graph_features)

    save_ml_dataset(aggregated, labels, output_dir)
    graph_features.to_csv(output_dir / "graph_features.csv", index=False)
    dataset.to_csv(output_dir / "ml_dataset.csv", index=False)

    graph_count = dataset["graph_id"].nunique()
    if graph_count < 5 or dataset["oracle_config_id"].nunique() < 2:
        summary = {
            "status": "preparation_only",
            "raw_rows": int(len(raw)),
            "graphs": int(graph_count),
            "ml_examples": int(len(dataset)),
            "reason": (
                "Smoke data is sufficient for contract and feature validation, "
                "but not for leakage-safe model training."
            ),
        }
        (output_dir / "smoke_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return

    assignments = split_graphs(dataset, seed=2027)
    split_dataset = attach_graph_splits(dataset, assignments)
    assignments.to_csv(output_dir / "graph_splits.csv", index=False)
    split_dataset.to_csv(output_dir / "ml_dataset_with_splits.csv", index=False)

    result = run_selector_experiment(split_dataset, aggregated, seed=2027)
    result.validation_metrics.to_csv(
        output_dir / "validation_model_comparison.csv", index=False
    )
    result.test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    result.baseline_predictions.to_csv(
        output_dir / "baseline_test_predictions.csv", index=False
    )
    joblib.dump(result.selected_model, output_dir / "selector_model.joblib")

    metrics = {
        "selected_model": result.selected_model_name,
        "test": result.test_metrics,
        "condition_global_best_baseline": result.baseline_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
