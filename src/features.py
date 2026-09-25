"""Leakage-safe graph features for the NQComp ML selector."""

from __future__ import annotations

import networkx as nx
import pandas as pd

from .graph_generation import GraphSpec, generate_graph
from .ml_dataset import BUDGETS


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

# Identifiers remain in saved tables for traceability but are excluded from X.
MODEL_FEATURE_COLUMNS = [
    "graph_family",
    "num_nodes",
    "num_edges",
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
    average_degree = sum(degrees) / len(degrees) if degrees else 0.0
    degree_std = float(pd.Series(degrees).std(ddof=0)) if len(degrees) > 1 else 0.0

    return {
        "graph_id": graph_id,
        "graph_family": graph_family,
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "graph_seed": graph_seed,
        "density": nx.density(graph),
        "average_degree": average_degree,
        "degree_std": degree_std,
        "min_degree": min(degrees) if degrees else 0,
        "max_degree": max(degrees) if degrees else 0,
        "triangle_count": sum(nx.triangles(graph).values()) // 3,
        "noise_condition": noise_condition,
        "resource_budget": resource_budget,
    }


def build_features_from_raw_results(results: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct frozen graphs and create one row per graph/noise/budget."""
    required = {
        "graph_id",
        "graph_family",
        "num_nodes",
        "num_edges",
        "graph_seed",
        "noise_condition",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Missing raw feature columns: {sorted(missing)}")

    graph_meta = results[
        ["graph_id", "graph_family", "num_nodes", "num_edges", "graph_seed"]
    ].drop_duplicates()
    if graph_meta["graph_id"].duplicated().any():
        raise ValueError("A graph_id has inconsistent graph metadata")

    rows: list[dict] = []
    for meta in graph_meta.itertuples(index=False):
        spec = GraphSpec(
            graph_id=str(meta.graph_id),
            family=str(meta.graph_family),
            num_nodes=int(meta.num_nodes),
            seed=int(meta.graph_seed),
            probability=0.35 if meta.graph_family == "erdos_renyi" else None,
            degree=4 if meta.graph_family == "random_regular" else None,
        )
        graph = generate_graph(spec)
        if graph.number_of_edges() != int(meta.num_edges):
            raise ValueError(
                f"Reconstructed edge count mismatch for {meta.graph_id}: "
                f"{graph.number_of_edges()} != {int(meta.num_edges)}"
            )

        noise_values = sorted(
            results.loc[results["graph_id"] == meta.graph_id, "noise_condition"]
            .dropna()
            .unique()
        )
        for noise_condition in noise_values:
            for resource_budget in BUDGETS:
                rows.append(
                    extract_graph_features(
                        graph=graph,
                        graph_id=str(meta.graph_id),
                        graph_family=str(meta.graph_family),
                        graph_seed=int(meta.graph_seed),
                        noise_condition=str(noise_condition),
                        resource_budget=resource_budget,
                    )
                )

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def build_graph_features(results: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate an already feature-enriched table."""
    missing = set(FEATURE_COLUMNS) - set(results.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")
    return (
        results[FEATURE_COLUMNS]
        .drop_duplicates(subset=["graph_id", "noise_condition", "resource_budget"])
        .reset_index(drop=True)
    )


def validate_no_leakage(features: pd.DataFrame) -> None:
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
    """Join pre-selection features to oracle labels one-to-one."""
    del aggregated  # Kept in the signature for compatibility with the handoff API.
    if labels.empty:
        raise ValueError("No oracle labels available")
    if graph_features.empty:
        raise ValueError("No graph features available")

    validate_no_leakage(graph_features)
    feature_key = ["graph_id", "noise_condition", "resource_budget"]
    if graph_features.duplicated(subset=feature_key).any():
        raise ValueError("Duplicate graph/noise/budget feature rows detected")

    dataset = graph_features.merge(
        labels,
        on=feature_key,
        how="inner",
        validate="one_to_one",
    )
    if len(dataset) != len(labels):
        raise ValueError("Some oracle labels do not have leakage-safe feature rows")
    return dataset
