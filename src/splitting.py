"""Leakage-safe graph-level train/validation/test splitting."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


SPLIT_FRACTIONS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}


def _target_counts(total: int) -> dict[str, int]:
    """Convert split fractions to exact counts using largest remainders."""
    raw = {name: total * fraction for name, fraction in SPLIT_FRACTIONS.items()}
    counts = {name: int(np.floor(value)) for name, value in raw.items()}
    remaining = total - sum(counts.values())

    order = sorted(
        SPLIT_FRACTIONS,
        key=lambda name: (-(raw[name] - counts[name]), list(SPLIT_FRACTIONS).index(name)),
    )
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def split_graphs(
    dataset: pd.DataFrame,
    seed: int = 2027,
) -> pd.DataFrame:
    """Assign each unique graph to one split, balanced by family and size.

    The assignment is performed before noise/budget row expansion. Exact global
    70/15/15 counts are enforced, while a deterministic greedy allocation keeps
    graph-family/size strata as balanced as the small stratum sizes permit.
    """
    required = {"graph_id", "graph_family", "num_nodes"}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")

    graph_rows = dataset[["graph_id", "graph_family", "num_nodes"]].drop_duplicates()
    if graph_rows["graph_id"].duplicated().any():
        raise ValueError("A graph_id has inconsistent family or node-count metadata")

    total = len(graph_rows)
    if total < 3:
        raise ValueError("At least three unique graphs are required for splitting")

    targets = _target_counts(total)
    remaining_capacity = targets.copy()
    rng = np.random.default_rng(seed)

    groups: dict[tuple[str, int], list[str]] = {}
    for key, group in graph_rows.groupby(["graph_family", "num_nodes"], sort=True):
        graph_ids = sorted(group["graph_id"].astype(str))
        rng.shuffle(graph_ids)
        groups[(str(key[0]), int(key[1]))] = graph_ids

    local_counts: dict[tuple[str, int], dict[str, int]] = {
        key: defaultdict(int) for key in groups
    }
    assignments: list[dict[str, str]] = []

    # Round-robin over strata avoids exhausting one split on the first strata.
    while any(groups.values()):
        for key in sorted(groups):
            if not groups[key]:
                continue
            graph_id = groups[key].pop()
            stratum_size = sum(local_counts[key].values()) + len(groups[key]) + 1

            candidates = [
                name for name, capacity in remaining_capacity.items() if capacity > 0
            ]
            if not candidates:
                raise RuntimeError("Split allocation exhausted capacity unexpectedly")

            def score(name: str) -> tuple[float, float, int]:
                local_target = stratum_size * SPLIT_FRACTIONS[name]
                local_deficit = local_target - local_counts[key][name]
                global_pressure = remaining_capacity[name] / max(targets[name], 1)
                priority = -list(SPLIT_FRACTIONS).index(name)
                return (local_deficit, global_pressure, priority)

            chosen = max(candidates, key=score)
            local_counts[key][chosen] += 1
            remaining_capacity[chosen] -= 1
            assignments.append({"graph_id": graph_id, "split": chosen})

    result = pd.DataFrame(assignments).sort_values("graph_id").reset_index(drop=True)
    actual = result["split"].value_counts().to_dict()
    if any(actual.get(name, 0) != count for name, count in targets.items()):
        raise RuntimeError(f"Split counts do not match targets: {actual} vs {targets}")
    return result


def attach_graph_splits(
    dataset: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Attach one graph-level assignment to every expanded ML example."""
    if assignments["graph_id"].duplicated().any():
        raise ValueError("Split assignments must contain one row per graph_id")

    result = dataset.merge(
        assignments[["graph_id", "split"]],
        on="graph_id",
        how="left",
        validate="many_to_one",
    )
    if result["split"].isna().any():
        missing = sorted(result.loc[result["split"].isna(), "graph_id"].unique())
        raise ValueError(f"Missing split assignments for graphs: {missing}")
    return result
