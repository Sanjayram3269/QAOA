"""Experiment orchestration and raw result serialization."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd

from .configurations import CONFIGURATIONS
from .maxcut import approximation_ratio, exact_maxcut
from .qaoa import evaluate_qaoa_parameters, run_qaoa

SCHEMA_VERSION = "1.0"


def _experiment_id(
    graph_id: str,
    config_id: str,
    noise_condition: str,
    run_seed: int,
) -> str:
    return f"{graph_id}_{config_id}_{noise_condition}_S{run_seed}"


def _make_record(
    result,
    graph: nx.Graph,
    graph_id: str,
    graph_family: str,
    graph_seed: int,
    noise_condition: str,
    config,
    optimum: int,
    run_seed: int,
) -> dict:
    """Convert a QAOA result into the canonical experiment schema."""

    record = asdict(result)

    record.pop("seed", None)
    shots = record.pop("shots")

    record.update(
        {
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
            "shots_per_circuit": shots,
            "exact_optimum": optimum,
            "run_seed": run_seed,
            "schema_version": SCHEMA_VERSION,
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

    record["circuit_executions"] = result.circuit_executions
    record["total_executed_shots"] = result.total_executed_shots

    return record


def evaluate_graph(
    graph: nx.Graph,
    graph_id: str,
    graph_family: str,
    graph_seed: int,
    noise_condition: str,
    run_seed_start: int | None = None,
    run_seeds: Iterable[int] | None = None,
    exact_max_nodes: int = 20,
    max_optimizer_evals: int = 80,
    max_circuit_executions: int = 200,
) -> pd.DataFrame:
    """
    Evaluate all canonical configurations.

    N0:
        Runs the normal QAOA optimizer.

    N1/N2:
        Also runs the normal optimizer when called directly. The experiment
        runner uses evaluate_noisy_graph() so noisy conditions can reuse
        parameters obtained from N0.
    """

    optimum, _ = exact_maxcut(
        graph,
        max_nodes=exact_max_nodes,
    )

    if run_seeds is not None:
        seeds = tuple(run_seeds)
    elif run_seed_start is not None:
    # One seed per graph/noise evaluation.
    # Each of the 12 canonical configurations gets the same seed.
        seeds = (run_seed_start,)
    else:
        raise ValueError(
            "Either run_seed_start or run_seeds must be provided."
        )

    rows: list[dict] = []

    for run_seed in seeds:
        for config in CONFIGURATIONS:

            result = run_qaoa(
                graph=graph,
                depth=config.depth,
                optimizer=config.optimizer,
                shots=config.shots,
                noise_condition=noise_condition,
                seed=run_seed,
                max_circuit_executions=max_circuit_executions,
            )

            rows.append(
                _make_record(
                    result,
                    graph,
                    graph_id,
                    graph_family,
                    graph_seed,
                    noise_condition,
                    config,
                    optimum,
                    run_seed,
                )
            )

    return pd.DataFrame(rows)


def evaluate_noisy_graph(
    graph: nx.Graph,
    graph_id: str,
    graph_family: str,
    graph_seed: int,
    noise_condition: str,
    n0_results: pd.DataFrame,
    exact_max_nodes: int = 20,
) -> pd.DataFrame:
    """
    Evaluate fixed parameters from N0 under a noisy condition.

    No optimizer is executed here. Each configuration requires exactly
    one simulator execution.
    """

    if noise_condition == "N0":
        raise ValueError(
            "evaluate_noisy_graph() is only for N1/N2."
        )

    optimum, _ = exact_maxcut(
        graph,
        max_nodes=exact_max_nodes,
    )

    rows: list[dict] = []

    for _, n0_row in n0_results.iterrows():

        config_id = str(n0_row["config_id"])

        config = next(
            c for c in CONFIGURATIONS
            if c.config_id == config_id
        )

        parameters = n0_row["optimal_parameters"]

        if isinstance(parameters, str):
            parameters = tuple(
                float(x.strip())
                for x in parameters.strip("()[]").split(",")
                if x.strip()
            )

        run_seed = int(n0_row["run_seed"])

        result = evaluate_qaoa_parameters(
            graph=graph,
            depth=config.depth,
            optimizer=config.optimizer,
            shots=config.shots,
            noise_condition=noise_condition,
            seed=run_seed,
            parameters=tuple(parameters),
        )

        rows.append(
            _make_record(
                result,
                graph,
                graph_id,
                graph_family,
                graph_seed,
                noise_condition,
                config,
                optimum,
                run_seed,
            )
        )

    return pd.DataFrame(rows)


def save_raw_results(
    results: pd.DataFrame,
    path: str | Path,
) -> None:
    """Persist raw experiment results without index."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)