"""Experiment orchestration and raw result serialization."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

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

    This function is also retained for direct noisy optimization if needed.
    The main N1/N2 experiment should use evaluate_noisy_graph(), which
    reuses parameters obtained from N0.
    """

    optimum, _ = exact_maxcut(
        graph,
        max_nodes=exact_max_nodes,
    )

    if run_seeds is not None:
        seeds = tuple(run_seeds)
    elif run_seed_start is not None:
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
                    result=result,
                    graph=graph,
                    graph_id=graph_id,
                    graph_family=graph_family,
                    graph_seed=graph_seed,
                    noise_condition=noise_condition,
                    config=config,
                    optimum=optimum,
                    run_seed=run_seed,
                )
            )

    return pd.DataFrame(rows)


def _parse_parameters(parameters) -> tuple[float, ...]:
    """Convert stored N0 optimal parameters into a numeric tuple."""

    if isinstance(parameters, str):
        cleaned = parameters.strip().strip("()[]")

        if not cleaned:
            return ()

        return tuple(
            float(x.strip())
            for x in cleaned.split(",")
            if x.strip()
        )

    if isinstance(parameters, (list, tuple)):
        return tuple(float(x) for x in parameters)

    raise TypeError(
        f"Unsupported optimal_parameters type: {type(parameters)}"
    )


def evaluate_noisy_graph(
    graph: nx.Graph,
    graph_id: str,
    graph_family: str,
    graph_seed: int,
    noise_condition: str,
    n0_results: pd.DataFrame,
    exact_max_nodes: int = 20,
    configs: Sequence | None = None,
    run_seeds: Iterable[int] | None = None,
) -> pd.DataFrame:
    """
    Evaluate the N0-optimized parameters under N1/N2 noise.

    IMPORTANT:
        No optimizer is executed here.

    For every N0 configuration supplied in n0_results, the exact same
    optimal parameters are executed once under the requested noise
    condition.

    Expected N0 input:
        12 rows for the current graph, one for each canonical configuration.

    Optional configs/run_seeds arguments are accepted for compatibility with
    older experiment-runner code, but the actual parameter source remains
    n0_results.
    """

    if noise_condition not in {"N1", "N2"}:
        raise ValueError(
            "evaluate_noisy_graph() is only for N1/N2."
        )

    if n0_results is None or n0_results.empty:
        raise ValueError(
            f"No N0 results supplied for graph {graph_id}."
        )

    # Safety: only use N0 results belonging to this graph.
    if "graph_id" in n0_results.columns:
        n0_results = n0_results[
            n0_results["graph_id"].astype(str) == str(graph_id)
        ].copy()

    if n0_results.empty:
        raise ValueError(
            f"No N0 rows found for graph {graph_id}."
        )

    # The canonical experiment has 12 configurations per graph.
    if configs is None:
        configs_by_id = {
            config.config_id: config
            for config in CONFIGURATIONS
        }
    else:
        configs_by_id = {
            config.config_id: config
            for config in configs
        }

    allowed_seeds = (
        set(int(seed) for seed in run_seeds)
        if run_seeds is not None
        else None
    )

    optimum, _ = exact_maxcut(
        graph,
        max_nodes=exact_max_nodes,
    )

    rows: list[dict] = []

    for _, n0_row in n0_results.iterrows():

        config_id = str(n0_row["config_id"])

        if config_id not in configs_by_id:
            raise ValueError(
                f"Unknown config_id '{config_id}' in N0 results."
            )

        config = configs_by_id[config_id]

        # Reuse only successful N0 results.
        if "run_status" in n0_row.index:
            if str(n0_row["run_status"]) != "success":
                continue

        run_seed = int(n0_row["run_seed"])

        if allowed_seeds is not None and run_seed not in allowed_seeds:
            continue

        parameters = _parse_parameters(
            n0_row["optimal_parameters"]
        )

        result = evaluate_qaoa_parameters(
            graph=graph,
            depth=config.depth,
            optimizer=config.optimizer,
            shots=config.shots,
            noise_condition=noise_condition,
            seed=run_seed,
            parameters=parameters,
        )

        rows.append(
            _make_record(
                result=result,
                graph=graph,
                graph_id=graph_id,
                graph_family=graph_family,
                graph_seed=graph_seed,
                noise_condition=noise_condition,
                config=config,
                optimum=optimum,
                run_seed=run_seed,
            )
        )

    return pd.DataFrame(rows)


def save_raw_results(
    results: pd.DataFrame,
    path: str | Path,
) -> None:
    """Persist raw experiment results without index."""

    output = Path(path)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output,
        index=False,
    )