"""Run the frozen, leakage-safe, resource-aware ML selector analysis.

The raw QAOA CSVs are never read or modified. This script consumes only the
derived common-graph ML table and writes a new final-selector output bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_selector import (  # noqa: E402
    FEATURE_SETS,
    SEED,
    TARGET,
    build_enhanced_graph_features,
    compare_models,
    fixed_baseline_choices,
    graph_clustered_paired_stats,
    make_model,
    merge_graph_features,
    row_metrics,
    score_test_selections,
    select_configurations,
    selector_permutation_importance,
    summarize_test,
    validate_split_integrity,
    validation_ablations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/ml/splits/ml_performance_common.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/ml/final_selector",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--permutation-repetitions", type=int, default=30)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_method_plot(summary: pd.DataFrame, path: Path) -> None:
    overall = summary.loc[summary["scope"] == "ALL"].iloc[0]
    labels = ["Random", "Fixed", "ML", "Oracle"]
    values = [
        overall["random_expected_mean"],
        overall["fixed_mean"],
        overall["ml_mean"],
        overall["oracle_mean"],
    ]
    colors = ["#9aa0a6", "#5f6368", "#1a73e8", "#34a853"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Mean expected approximation ratio")
    ax.set_title("Frozen test-set selector comparison")
    ax.set_ylim(max(0, min(values) - 0.025), min(1.0, max(values) + 0.02))
    ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_importance_plot(importance: pd.DataFrame, path: Path) -> None:
    shown = importance.sort_values("mean_regret_increase").tail(12)
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    ax.barh(
        shown["feature_group"],
        shown["mean_regret_increase"],
        xerr=shown["std_regret_increase"],
        color="#1a73e8",
        alpha=0.88,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Increase in validation selection regret after permutation")
    ax.set_title("Selector-aware grouped permutation importance")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input)
    validate_split_integrity(data)
    enhanced = build_enhanced_graph_features(data)
    data = merge_graph_features(data, enhanced)

    train = data[data["split"] == "train"].copy()
    validation = data[data["split"] == "validation"].copy()
    test = data[data["split"] == "test"].copy()
    full_features = FEATURE_SETS["full"]

    comparison, selected_name, fitted = compare_models(
        train, validation, full_features, args.seed
    )
    ablations = validation_ablations(selected_name, train, validation, args.seed)
    importance = selector_permutation_importance(
        fitted[selected_name],
        validation,
        full_features,
        args.seed,
        args.permutation_repetitions,
    )

    # The model family and feature set are now frozen. Test data are touched
    # only after this point and are never used to revise those choices.
    train_validation = pd.concat([train, validation], ignore_index=True)
    final_model = make_model(selected_name, full_features, args.seed)
    final_model.fit(train_validation[full_features.columns], train_validation[TARGET])
    test_predictions = final_model.predict(test[full_features.columns])
    test_row_metrics = row_metrics(test[TARGET], test_predictions)

    selections = select_configurations(final_model, test, full_features)
    fixed_choices = fixed_baseline_choices(train)
    scored = score_test_selections(selections, test, fixed_choices)
    summary = summarize_test(scored)
    paired_stats = graph_clustered_paired_stats(
        scored, args.seed, args.bootstrap_repetitions
    )

    outputs = {
        "enhanced_graph_features.csv": enhanced,
        "validation_model_comparison.csv": comparison,
        "validation_feature_ablations.csv": ablations,
        "validation_permutation_importance.csv": importance,
        "fixed_baseline_choices.csv": fixed_choices,
        "test_selected_configurations.csv": scored,
        "test_summary.csv": summary,
        "test_graph_clustered_statistics.csv": paired_stats,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)

    if not args.no_plots:
        save_method_plot(summary, args.output_dir / "test_method_comparison.svg")
        save_importance_plot(
            importance, args.output_dir / "validation_permutation_importance.svg"
        )

    overall = summary.loc[summary["scope"] == "ALL"].iloc[0]
    manifest = {
        "analysis": "resource-aware QAOA configuration selector",
        "seed": args.seed,
        "input": str(args.input.relative_to(ROOT)),
        "input_sha256": sha256(args.input),
        "raw_qaoa_files_modified": False,
        "target": TARGET,
        "split_unit": "graph_id",
        "split_graph_counts": {
            "train": int(train["graph_id"].nunique()),
            "validation": int(validation["graph_id"].nunique()),
            "test": int(test["graph_id"].nunique()),
        },
        "budget_definition": {
            "B256": "configuration shots per circuit <= 256",
            "B512": "configuration shots per circuit <= 512",
        },
        "model_selection_partition": "validation only",
        "model_selection_metric": "mean selection regret across validation graph/noise/budget cases",
        "selected_model": selected_name,
        "selected_feature_set": full_features.columns,
        "final_fit_partition": "train + validation",
        "test_evaluation_policy": "one frozen evaluation after model and features were selected",
        "test_row_prediction_metrics": test_row_metrics,
        "test_selector_metrics": {
            "ml_mean": float(overall["ml_mean"]),
            "fixed_mean": float(overall["fixed_mean"]),
            "random_expected_mean": float(overall["random_expected_mean"]),
            "oracle_mean": float(overall["oracle_mean"]),
            "ml_minus_fixed": float(overall["ml_minus_fixed"]),
            "relative_gain_percent": float(overall["relative_gain_percent"]),
            "mean_regret": float(overall["mean_regret"]),
            "oracle_selection_accuracy": float(
                overall["oracle_selection_accuracy"]
            ),
            "oracle_top3_accuracy": float(overall["oracle_top3_accuracy"]),
        },
        "inference_unit": "graph_id",
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "permutation_importance_repetitions": args.permutation_repetitions,
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(comparison.to_string(index=False))
    print(f"\nSelected on validation regret: {selected_name}")
    print("\nFrozen test summary:")
    print(summary.loc[summary["scope"] == "ALL"].to_string(index=False))
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
