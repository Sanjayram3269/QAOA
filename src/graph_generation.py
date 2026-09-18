"""Deterministic Max-Cut graph generation and manifest utilities."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd


@dataclass(frozen=True)
class GraphSpec:
    graph_id: str
    family: str
    num_nodes: int
    seed: int
    probability: float | None = None
    degree: int | None = None


def generate_graph(spec: GraphSpec) -> nx.Graph:
    """Generate one deterministic graph from a GraphSpec."""
    if spec.family == "erdos_renyi":
        if spec.probability is None:
            raise ValueError("Erdos-Renyi graphs require probability")
        graph = nx.gnp_random_graph(
            spec.num_nodes,
            spec.probability,
            seed=spec.seed,
        )
    elif spec.family == "random_regular":
        if spec.degree is None:
            raise ValueError("Random-regular graphs require degree")
        if spec.degree >= spec.num_nodes:
            raise ValueError("Random-regular degree must be less than node count")
        if (spec.num_nodes * spec.degree) % 2:
            raise ValueError("n * degree must be even for a regular graph")
        graph = nx.random_regular_graph(
            spec.degree,
            spec.num_nodes,
            seed=spec.seed,
        )
    else:
        raise ValueError(f"Unsupported graph family: {spec.family}")

    # Keep node labels canonical and stable for the data contract.
    return nx.convert_node_labels_to_integers(graph, ordering="sorted")


def graph_metadata(graph: nx.Graph, spec: GraphSpec) -> dict:
    """Return stable graph-level metadata used by downstream pipelines."""
    degrees = [degree for _, degree in graph.degree()]
    return {
        "graph_id": spec.graph_id,
        "graph_family": spec.family,
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "graph_seed": spec.seed,
        "degree_min": min(degrees) if degrees else 0,
        "degree_max": max(degrees) if degrees else 0,
        "degree_mean": sum(degrees) / len(degrees) if degrees else 0.0,
        "density": nx.density(graph),
    }


def build_manifest(
    families: Iterable[str] = ("erdos_renyi", "random_regular"),
    node_counts: Iterable[int] = (10, 12, 15, 18, 20),
    instances_per_setting: int = 10,
    er_probability: float = 0.35,
    random_regular_degree: int = 4,
    seed_start: int = 10000,
) -> pd.DataFrame:
    """Create the authoritative graph manifest for an experiment batch."""
    rows: list[dict] = []
    seed = seed_start
    counter = 1

    for family in families:
        for n in node_counts:
            for _ in range(instances_per_setting):
                spec = GraphSpec(
                    graph_id=f"G{counter:04d}",
                    family=family,
                    num_nodes=n,
                    seed=seed,
                    probability=er_probability if family == "erdos_renyi" else None,
                    degree=random_regular_degree if family == "random_regular" else None,
                )
                graph = generate_graph(spec)
                rows.append({**asdict(spec), **graph_metadata(graph, spec)})
                counter += 1
                seed += 1

    return pd.DataFrame(rows)


def save_manifest(manifest: pd.DataFrame, path: str | Path) -> None:
    """Save a manifest without modifying its contents."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)


if __name__ == "__main__":
    manifest = build_manifest()
    save_manifest(manifest, "data/graphs/master_graph_manifest.csv")
    print(f"Wrote {len(manifest)} graph specifications.")
