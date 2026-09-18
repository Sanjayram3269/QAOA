"""Exact Max-Cut utilities for small research graphs."""

from __future__ import annotations

from typing import Iterable

import networkx as nx


def cut_value(graph: nx.Graph, partition: Iterable[int]) -> int:
    """Return the number of edges crossing a binary partition.

    `partition[node]` must be 0 or 1 for every node.
    """
    labels = list(partition)
    if len(labels) != graph.number_of_nodes():
        raise ValueError("Partition length must equal number of graph nodes")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Partition labels must be 0 or 1")

    return sum(labels[u] != labels[v] for u, v in graph.edges())


def exact_maxcut(graph: nx.Graph, max_nodes: int = 20) -> tuple[int, list[int]]:
    """Solve Max-Cut exactly by enumerating one representative per complement pair.

    For n nodes, fixing node 0 to side 0 reduces enumeration from 2^n to
    2^(n-1). The method is intended for the project's small graph sizes.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return 0, []
    if n > max_nodes:
        raise ValueError(
            f"Exact solver is limited to {max_nodes} nodes; received {n}."
        )

    best_value = -1
    best_partition: list[int] = [0] * n

    # Node 0 is fixed to side 0 because a cut and its complement have the same value.
    for mask in range(1 << (n - 1)):
        partition = [0] * n
        for node in range(1, n):
            partition[node] = (mask >> (node - 1)) & 1

        value = cut_value(graph, partition)
        if value > best_value:
            best_value = value
            best_partition = partition

    return best_value, best_partition


def approximation_ratio(cut: float, optimum: float) -> float:
    """Return cut/optimum, with a defined value for the zero-optimum case."""
    if optimum == 0:
        return 1.0 if cut == 0 else 0.0
    return float(cut) / float(optimum)
