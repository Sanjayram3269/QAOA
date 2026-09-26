"""Build the first ML-ready dataset from the frozen raw QAOA CSVs.

This script NEVER modifies the raw files in data/results/.
It aggregates the two run seeds for each graph/configuration/condition and
creates graph-level structural features plus configuration-level performance.

Inputs:
    data/results/n0_results.csv
    data/results/noisy_results.csv

Outputs:
    data/ml/ml_performance.csv
    data/ml/graph_features.csv

The initial ML formulation is performance prediction:
    graph features + configuration + noise condition -> mean expected approximation ratio

The selector can later evaluate all 12 configurations for a graph and choose
the one with the highest predicted performance.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
OUT = ROOT / "data" / "ml"
OUT.mkdir(parents=True, exist_ok=True)

N0_PATH = RESULTS / "n0_results.csv"
NOISE_PATH = RESULTS / "noisy_results.csv"


def first_existing_column(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of these columns were found: {candidates}\nAvailable: {list(df.columns)}")
    return None


def load_raw():
    if not N0_PATH.exists() or not NOISE_PATH.exists():
        raise FileNotFoundError(
            "Expected raw files are missing. Required:\n"
            f"  {N0_PATH}\n  {NOISE_PATH}"
        )
    n0 = pd.read_csv(N0_PATH)
    noise = pd.read_csv(NOISE_PATH)
    n0["source_file"] = "n0_results.csv"
    noise["source_file"] = "noisy_results.csv"
    return pd.concat([n0, noise], ignore_index=True)


def density(n, m):
    return 0.0 if n <= 1 else (2.0 * m) / (n * (n - 1))


def build_graph_features(df):
    # These are graph features already represented in the raw experiment.
    # We intentionally do not use QAOA outcome columns as graph features.
    node_col = first_existing_column(df, ["num_nodes", "n_nodes", "nodes"])
    edge_col = first_existing_column(df, ["num_edges", "n_edges", "edges"])
    family_col = first_existing_column(df, ["graph_family", "family"])
    graph_seed_col = first_existing_column(df, ["graph_seed", "seed_graph"], required=False)

    gcols = ["graph_id", node_col, edge_col, family_col]
    if graph_seed_col:
        gcols.append(graph_seed_col)

    graphs = df[gcols].drop_duplicates("graph_id").copy()
    if len(graphs) != df["graph_id"].nunique():
        raise ValueError("Graph metadata is inconsistent: multiple structural records per graph_id.")

    graphs["density"] = [density(n, m) for n, m in zip(graphs[node_col], graphs[edge_col])]
    graphs = graphs.rename(columns={node_col: "num_nodes", edge_col: "num_edges", family_col: "graph_family"})
    if graph_seed_col:
        graphs = graphs.rename(columns={graph_seed_col: "graph_seed"})
    return graphs


def build_performance(df):
    required = ["graph_id", "config_id", "noise_condition", "run_seed"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required identity columns: {missing}")

    ratio_col = first_existing_column(
        df,
        ["expected_approximation_ratio", "expected_approx_ratio"],
    )
    depth_col = first_existing_column(df, ["depth", "qaoa_depth"])
    optimizer_col = first_existing_column(df, ["optimizer"])
    shots_col = first_existing_column(df, ["shots_per_circuit", "shots"])

    keep = [
        "graph_id", "config_id", "noise_condition", "run_seed",
        depth_col, optimizer_col, shots_col, ratio_col,
    ]
    x = df[keep].copy().rename(
        columns={depth_col: "depth", optimizer_col: "optimizer", shots_col: "shots",
                 ratio_col: "expected_approximation_ratio"}
    )

    x["expected_approximation_ratio"] = pd.to_numeric(x["expected_approximation_ratio"], errors="coerce")
    if x["expected_approximation_ratio"].isna().any():
        raise ValueError("Missing/non-numeric expected approximation ratios found.")

    # Preserve the raw seed-level rows in a separate audit file.
    x.to_csv(OUT / "ml_seed_level_performance.csv", index=False)

    grouped = (
        x.groupby(["graph_id", "config_id", "noise_condition", "depth", "optimizer", "shots"], as_index=False)
         .agg(
             mean_expected_approximation_ratio=("expected_approximation_ratio", "mean"),
             std_expected_approximation_ratio=("expected_approximation_ratio", "std"),
             n_seeds=("expected_approximation_ratio", "count"),
         )
    )
    grouped["std_expected_approximation_ratio"] = grouped["std_expected_approximation_ratio"].fillna(0.0)

    return grouped


def main():
    print("=" * 72)
    print("NQComp 2027 — BUILD ML DATASET")
    print("=" * 72)

    raw = load_raw()
    print(f"Raw rows loaded: {len(raw):,}")
    print(f"Raw columns: {len(raw.columns)}")

    graphs = build_graph_features(raw)
    performance = build_performance(raw)

    # Join structural information onto every graph/configuration/condition row.
    ml = performance.merge(graphs, on="graph_id", how="left", validate="many_to_one")

    # Explicit categorical encodings for reproducible modeling later.
    ml["optimizer_code"] = ml["optimizer"].map({"COBYLA": 0, "SPSA": 1})
    ml["noise_code"] = ml["noise_condition"].map({"N0": 0, "N1": 1, "N2": 2})
    ml["family_code"] = pd.Categorical(ml["graph_family"]).codes

    if ml[["optimizer_code", "noise_code"]].isna().any().any():
        raise ValueError("Unknown optimizer/noise category encountered.")

    # Sort for deterministic output.
    ml = ml.sort_values(["noise_condition", "graph_id", "config_id"]).reset_index(drop=True)
    graphs = graphs.sort_values("graph_id").reset_index(drop=True)

    # A compact manifest records exactly what was used to build this derived dataset.
    manifest = {
        "raw_files": ["data/results/n0_results.csv", "data/results/noisy_results.csv"],
        "raw_rows": int(len(raw)),
        "graph_count": int(raw["graph_id"].nunique()),
        "performance_rows": int(len(ml)),
        "config_count": int(ml["config_id"].nunique()),
        "noise_conditions": sorted(ml["noise_condition"].unique().tolist()),
        "target": "mean_expected_approximation_ratio",
        "seed_aggregation": "mean over run_seed; std retained",
        "target_graph_outcomes_used_as_features": False,
        "raw_files_modified": False,
    }

    graphs.to_csv(OUT / "graph_features.csv", index=False)
    ml.to_csv(OUT / "ml_performance.csv", index=False)
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("OUTPUT")
    print("=" * 72)
    print(f"Graph features : {OUT / 'graph_features.csv'}")
    print(f"ML performance : {OUT / 'ml_performance.csv'}")
    print(f"Seed audit     : {OUT / 'ml_seed_level_performance.csv'}")
    print(f"Manifest       : {OUT / 'dataset_manifest.json'}")
    print(f"Graphs         : {graphs['graph_id'].nunique()}")
    print(f"ML rows        : {len(ml):,}")
    print(f"Configs        : {ml['config_id'].nunique()}")
    print(f"Noise          : {sorted(ml['noise_condition'].unique())}")
    print("\n✅ ML dataset created. Raw experimental CSVs were not modified.")


if __name__ == "__main__":
    main()
