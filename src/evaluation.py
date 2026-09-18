"""Experiment orchestration and raw result serialization."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd

from .configurations import CONFIGURATIONS
from .maxcut import approximation_ratio, exact_maxcut
from .qaoa import run_qaoa

SCHEMA_VERSION = "1.0"


def _experiment_id(
    graph_id: str,
    config_id: str,
    noise_condition: str,
    run_seed: int,
) -> str:
    return f"{graph_id}_{config_id}_{noise_condition}_S{run_seed}"


def evaluate_graph(
    graph: nx.Graph,
    graph_id: str,
    graph_family: str,
    graph_seed: int,
    noise_condition: str,
    run_seeds: Iterable[int],
    exact_max_nodes: int = 20,
    max_circuit_executions: int = 200,
) -> pd.DataFrame:
    """Evaluate every canonical configuration across repeated paired seeds."""
    optimum, _ = exact_maxcut(graph, max_nodes=exact_max_nodes)
    rows: list[dict] = []

    for run_seed in tuple(run_seeds):
        for config in CONFIGURATIONS:
            common = {
                "experiment_id": _experiment_id(
                    graph_id,
                    config.config_id,
                    noise_condition,
                    run_seed,
                ),
                "graph_id": graph_id,
                "graph_family": graph_family,
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "graph_seed": graph_seed,
                "noise_condition": noise_condition,
                "config_id": config.config_id,
                "depth": config.depth,
                "optimizer": config.optimizer,
                "shots_per_circuit": config.shots,
                "exact_optimum": optimum,
                "run_seed": run_seed,
                "schema_version": SCHEMA_VERSION,
            }

            try:
                result = run_qaoa(
                    graph=graph,
                    depth=config.depth,
                    optimizer=config.optimizer,
                    shots=config.shots,
                    noise_condition=noise_condition,
                    seed=run_seed,
                    max_circuit_executions=max_circuit_executions,
                )
                record = asdict(result)
                record.pop("seed")
                record.pop("shots")
                record.update(
                    {
                        **common,
                        "expected_approximation_ratio": approximation_ratio(
                            result.expected_cut,
                            optimum,
                        ),
                        "best_sampled_approximation_ratio": approximation_ratio(
                            result.best_sampled_cut,
                            optimum,
                        ),
                        "run_status": "success",
                        "failure_reason": "",
                    }
                )
            except Exception as exc:
                record = {
                    **common,
                    "expected_cut": None,
                    "best_sampled_cut": None,
                    "expected_approximation_ratio": None,
                    "best_sampled_approximation_ratio": None,
                    "optimizer_evaluations": None,
                    "circuit_executions": None,
                    "total_executed_shots": None,
                    "two_qubit_gates": None,
                    "circuit_depth": None,
                    "simulator_runtime_seconds": None,
                    "optimal_parameters": None,
                    "run_status": "failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }

            rows.append(record)

    return pd.DataFrame(rows)


def save_raw_results(results: pd.DataFrame, path: str | Path) -> None:
    """Persist raw experiment results without index."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
