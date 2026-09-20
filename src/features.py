"""Leakage-safe graph features for the NQComp ML selector."""

from __future__ import annotations

import networkx as nx
import pandas as pd

from .ml_dataset import BUDGETS


# These are the only feature columns allowed as ML inputs.
FEATURE_COLUMNS = [
    "graph_id",
    "graph_family",
    "num_nodes",
    "num_edges",
    "graph_seed",
    "density",
    "average_degree",
    "degree_std",
    "min_degree",
    "max_degree",
    "triangle_count",
    "noise_condition",
    "resource_budget",
]


def extract_graph_features(
    graph: nx.Graph,
    graph_id: str,
    graph_family: str,
    graph_seed: int,
    noise_condition: str,
    resource_budget: str,
) -> dict:
    """Extract graph and known-condition features before QAOA selection."""

    degrees = [degree for _, degree in graph.degree()]

    if degrees:
        average_degree = sum(degrees) / len(degrees)

        if len(degrees) > 1:
            degree_std = float(
                pd.Series(degrees).std(ddof=0)
            )
        else:
            degree_std = 0.0

        min_degree = min(degrees)
        max_degree = max(degrees)
    else:
        average_degree = 0.0
        degree_std = 0.0
        min_degree = 0
        max_degree = 0

    triangle_count = (
        sum(nx.triangles(graph).values()) // 3
    )

    return {
        "graph_id": graph_id,
        "graph_family": graph_family,
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "graph_seed": graph_seed,
        "density": nx.density(graph),
        "average_degree": average_degree,
        "degree_std": degree_std,
        "min_degree": min_degree,
        "max_degree": max_degree,
        "triangle_count": triangle_count,
        "noise_condition": noise_condition,
        "resource_budget": resource_budget,
    }


def build_graph_features(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one feature row per graph/noise/budget.

    This function expects graph-level structural features to already
    be present in the input DataFrame.
    """

    required = set(FEATURE_COLUMNS)

    missing = required - set(results.columns)

    if missing:
        raise ValueError(
            f"Missing required feature columns: {sorted(missing)}"
        )

    feature_rows = (
        results[FEATURE_COLUMNS]
        .drop_duplicates(
            subset=[
                "graph_id",
                "graph_family",
                "num_nodes",
                "num_edges",
                "graph_seed",
                "noise_condition",
                "resource_budget",
            ]
        )
        .reset_index(drop=True)
    )

    return feature_rows


def validate_no_leakage(
    features: pd.DataFrame,
) -> None:
    """Ensure post-execution/QAOA information is absent."""

    forbidden = {
        "config_id",
        "depth",
        "optimizer",
        "shots_per_circuit",
        "expected_cut",
        "best_sampled_cut",
        "exact_optimum",
        "expected_approximation_ratio",
        "best_sampled_approximation_ratio",
        "optimizer_evaluations",
        "circuit_executions",
        "total_executed_shots",
        "two_qubit_gates",
        "circuit_depth",
        "simulator_runtime_seconds",
        "optimal_parameters",
        "oracle_config_id",
        "oracle_utility",
        "oracle_quality",
        "oracle_cost",
    }

    leaked = forbidden.intersection(features.columns)

    if leaked:
        raise ValueError(
            "Data leakage detected. Forbidden columns present: "
            f"{sorted(leaked)}"
        )


def build_ml_features(
    aggregated: pd.DataFrame,
    labels: pd.DataFrame,
    graph_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the final leakage-safe ML dataset.

    Parameters
    ----------
    aggregated:
        Configuration-level QAOA aggregate results.

    labels:
        Oracle labels generated from the frozen utility.

    graph_features:
        Graph structural features generated before configuration selection.
    """

    if labels.empty:
        raise ValueError("No oracle labels available")

    if graph_features.empty:
        raise ValueError("No graph features available")

    # Validate graph features before joining anything with labels.
    validate_no_leakage(graph_features)

    features = graph_features.copy()

    # Make sure each graph/noise/budget combination occurs once.
    feature_key = [
        "graph_id",
        "noise_condition",
        "resource_budget",
    ]

    if features.duplicated(subset=feature_key).any():
        raise ValueError(
            "Duplicate graph/noise/budget feature rows detected"
        )

    # Check that every labelled example has a corresponding
    # leakage-safe feature row.
    labels_key = labels[feature_key].drop_duplicates()

    missing = labels_key.merge(
        features[feature_key],
        on=feature_key,
        how="left",
        indicator=True,
    )

    missing = missing[
        missing["_merge"] == "left_only"
    ]

    if not missing.empty:
        raise ValueError(
            "Missing graph features for labelled rows: "
            f"{missing[feature_key].to_dict('records')}"
        )

    # Final ML table.
    dataset = features.merge(
        labels,
        on=feature_key,
        how="inner",
        validate="one_to_one",
    )

    return dataset