"""Create deterministic, leakage-safe graph-level ML splits for NQComp 2027.

The split unit is the graph, never an individual row. This prevents the same
problem graph from appearing in train/validation/test through different QAOA
configurations or noise conditions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "ml" / "ml_performance.csv"
OUT = ROOT / "data" / "ml" / "splits"
SEED = 2027


def split_graphs(graph_ids: list[str], seed: int = SEED) -> tuple[list[str], list[str], list[str]]:
    """Deterministically split graph IDs 70/15/15."""
    ids = np.array(sorted(graph_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(0.70 * n))
    n_val = int(round(0.15 * n))
    train = sorted(ids[:n_train].tolist())
    val = sorted(ids[n_train:n_train + n_val].tolist())
    test = sorted(ids[n_train + n_val:].tolist())
    return train, val, test


def main() -> None:
    print("=" * 72)
    print("NQComp 2027 — CREATE LEAKAGE-SAFE ML SPLITS")
    print("=" * 72)

    if not INPUT.exists():
        raise FileNotFoundError(f"Missing ML dataset: {INPUT}")

    df = pd.read_csv(INPUT)
    required = {"graph_id", "noise_condition", "config_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Noise-aware experiment: only graphs with complete N0, N1 and N2 coverage.
    coverage = df.groupby("graph_id")["noise_condition"].agg(lambda s: set(s))
    complete_noisy = sorted(coverage[coverage.apply(lambda x: {"N0", "N1", "N2"}.issubset(x))].index.tolist())

    # Require all 12 configurations for every condition in the common population.
    counts = (
        df[df["graph_id"].isin(complete_noisy)]
        .groupby(["graph_id", "noise_condition"])["config_id"]
        .nunique()
    )
    bad = counts[counts != 12]
    if not bad.empty:
        raise ValueError(f"Incomplete graph/condition configuration coverage:\n{bad}")

    train, val, test = split_graphs(complete_noisy)
    sets = {"train": train, "validation": val, "test": test}

    # Hard overlap checks.
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))
    assert set(train) | set(val) | set(test) == set(complete_noisy)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, ids in sets.items():
        pd.DataFrame({"graph_id": ids}).to_csv(OUT / f"{name}_graphs.csv", index=False)

    split_map = {gid: name for name, ids in sets.items() for gid in ids}
    split_df = pd.DataFrame({"graph_id": complete_noisy})
    split_df["split"] = split_df["graph_id"].map(split_map)
    split_df.to_csv(OUT / "graph_split_assignments.csv", index=False)

    subset = df[df["graph_id"].isin(complete_noisy)].copy()
    subset["split"] = subset["graph_id"].map(split_map)
    subset.to_csv(OUT / "ml_performance_common.csv", index=False)

    row_counts = {}
    for name in ("train", "validation", "test"):
        part = subset[subset["split"] == name]
        row_counts[name] = {
            "graphs": int(part["graph_id"].nunique()),
            "rows": int(len(part)),
            "N0": int((part["noise_condition"] == "N0").sum()),
            "N1": int((part["noise_condition"] == "N1").sum()),
            "N2": int((part["noise_condition"] == "N2").sum()),
        }

    manifest = {
        "random_seed": SEED,
        "split_strategy": "graph-level 70/15/15",
        "population": "graphs with complete N0, N1, N2 coverage and 12 configs per condition",
        "total_common_graphs": len(complete_noisy),
        "train_graphs": train,
        "validation_graphs": val,
        "test_graphs": test,
        "row_counts": row_counts,
        "graph_overlap": {"train_validation": 0, "train_test": 0, "validation_test": 0},
    }
    (OUT / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Common complete graphs: {len(complete_noisy)}")
    print(f"Train:      {len(train)} graphs")
    print(f"Validation: {len(val)} graphs")
    print(f"Test:       {len(test)} graphs")
    print("\nRows by split:")
    for name, info in row_counts.items():
        print(f"  {name:10s}: {info['rows']} rows | N0={info['N0']} N1={info['N1']} N2={info['N2']}")
    print(f"\nOutput: {OUT}")
    print("\n✅ Graph-level split created with zero graph overlap.")


if __name__ == "__main__":
    main()
