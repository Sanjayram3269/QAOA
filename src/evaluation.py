"""Experiment orchestration and raw result serialization."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import networkx as nx
import pandas as pd

from .configurations import CONFIGURATIONS
from .maxcut import approximation_ratio, exact_maxcut
from .qaoa import run_qaoa


def evaluate_graph(
    graph: nx.Graph,
    graph_id: str,
    graph_family: str,
    graph_seed: int,
    noise_condition: str,
    run_seed_start: int,
    exact_max_nodes: int = 20,
    max_optimizer_evals: int = 80,
) -> pd.DataFrame:
    """Evaluate every canonical QAOA configuration for one graph/condition."""
    optimum, _ = exact_maxcut(graph, max_nodes=exact_max_nodes)
    rows: list[dict] = []

    for offset, config in enumerate(CONFIGURATIONS):
        result = run_qaoa(
            graph=graph,
            depth=config.depth,
            optimizer=config.optimizer,
            shots=config.shots,
            noise_condition=noise_condition,
            seed=run_seed_start + offset,
            max_optimizer_evals=max_optimizer_evals,
        )
        record = asdict(result)
        record["run_seed"] = record.pop("seed")
        record.update(
            {
                "graph_id": graph_id,
                "graph_family": graph_family,
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "graph_seed": graph_seed,
                "noise_condition": noise_condition,
                "config_id": config.config_id,
                "exact_optimum": optimum,
                "expected_approximation_ratio": approximation_ratio(
                    result.expected_cut, optimum
                ),
                "best_sampled_approximation_ratio": approximation_ratio(
                    result.best_sampled_cut, optimum
                ),
            }
        )
        rows.append(record)

    return pd.DataFrame(rows)


def save_raw_results(results: pd.DataFrame, path: str | Path) -> None:
    """Persist raw experiment results without index."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
