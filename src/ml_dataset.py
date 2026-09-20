"""Build ML-ready labels from validated QAOA raw results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ALPHA = 0.8
COST_NORMALIZER = 102400

BUDGETS = {
    "B256": 256,
    "B512": 512,
}


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate successful paired runs across the three seeds."""

    successful = results[results["run_status"] == "success"].copy()

    if successful.empty:
        raise ValueError("No successful QAOA results available")

    group_cols = [
        "graph_id",
        "graph_family",
        "num_nodes",
        "num_edges",
        "graph_seed",
        "noise_condition",
        "config_id",
        "depth",
        "optimizer",
        "shots_per_circuit",
    ]

    aggregated = (
        successful.groupby(group_cols, as_index=False)
        .agg(
            mean_expected_approximation_ratio=(
                "expected_approximation_ratio",
                "mean",
            ),
            mean_best_sampled_approximation_ratio=(
                "best_sampled_approximation_ratio",
                "mean",
            ),
            mean_total_executed_shots=(
                "total_executed_shots",
                "mean",
            ),
            successful_runs=("run_seed", "count"),
        )
    )

    return aggregated


def build_labels(aggregated: pd.DataFrame) -> pd.DataFrame:
    """Create B256/B512 oracle labels using the frozen utility."""

    rows = []

    for (graph_id, noise_condition), group in aggregated.groupby(
        ["graph_id", "noise_condition"]
    ):
        for budget_name, budget in BUDGETS.items():

            feasible = group[
                group["shots_per_circuit"] <= budget
            ].copy()

            if feasible.empty:
                continue

            feasible["Q_norm"] = (
                feasible["mean_expected_approximation_ratio"]
            )

            feasible["C_norm"] = (
                feasible["mean_total_executed_shots"]
                / COST_NORMALIZER
            ).clip(upper=1.0)

            feasible["utility"] = (
                ALPHA * feasible["Q_norm"]
                - (1.0 - ALPHA) * feasible["C_norm"]
            )

            max_utility = feasible["utility"].max()

            tied = feasible[
                feasible["utility"] >= max_utility - 0.005
            ].copy()

            # Frozen tie-breaking rules.
            tied = tied.sort_values(
                [
                    "mean_total_executed_shots",
                    "depth",
                    "shots_per_circuit",
                    "config_id",
                ]
            )

            winner = tied.iloc[0]

            rows.append(
                {
                    "graph_id": graph_id,
                    "noise_condition": noise_condition,
                    "resource_budget": budget_name,
                    "oracle_config_id": winner["config_id"],
                    "oracle_utility": winner["utility"],
                    "oracle_quality": winner[
                        "mean_expected_approximation_ratio"
                    ],
                    "oracle_cost": winner[
                        "mean_total_executed_shots"
                    ],
                }
            )

    return pd.DataFrame(rows)


def build_ml_dataset(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aggregated experiment data and oracle ML labels."""

    aggregated = aggregate_results(results)
    labels = build_labels(aggregated)

    return aggregated, labels


def save_ml_dataset(
    aggregated: pd.DataFrame,
    labels: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Save intermediate aggregated data and final oracle labels."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregated.to_csv(
        output_dir / "aggregated_results.csv",
        index=False,
    )

    labels.to_csv(
        output_dir / "oracle_labels.csv",
        index=False,
    )