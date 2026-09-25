"""
Run the NQComp 2027 QAOA experiment in fast, resumable batches.

Modes
-----
n0
    Run the 12 canonical QAOA configurations with two run seeds per graph.

noise
    Reuse the N0 optimal parameters and evaluate them under N1 and N2.
    No optimizer is run during N1/N2.

Important resumability behavior
-------------------------------
- Results are saved AFTER EVERY GRAPH, not only at the end of a batch.
- Existing completed graph/config/noise/seed rows are not rerun.
- If a graph is partially complete, only the missing N1/N2 rows are run.
- Therefore a Colab/session interruption does not destroy completed work
  as long as the CSV is persisted outside the ephemeral runtime.
- --graph-ids can be used for explicit recovery.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Make repository root importable when running:
# python scripts/run_experiment.py
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import (
    evaluate_graph,
    evaluate_noisy_graph,
    save_raw_results,
)
from src.graph_generation import GraphSpec, generate_graph


# ===========================================================================
# EXPERIMENT SETTINGS
# ===========================================================================

N0_GRAPHS = 100
NOISY_GRAPHS = 100

DEFAULT_BATCH_SIZE = 10

NOISE_CONDITIONS = {
    "N0": False,
    "N1": True,
    "N2": True,
}

OUTPUT_DIR = ROOT / "data" / "results"

N0_OUTPUT = OUTPUT_DIR / "n0_results.csv"
NOISY_OUTPUT = OUTPUT_DIR / "noisy_results.csv"

MANIFEST_PATH = ROOT / "data" / "graphs" / "master_graph_manifest.csv"

# The 12 canonical configurations.
CONFIGS = [
    ("C01", 1, "COBYLA", 256),
    ("C02", 1, "COBYLA", 512),
    ("C03", 1, "SPSA", 256),
    ("C04", 1, "SPSA", 512),

    ("C05", 2, "COBYLA", 256),
    ("C06", 2, "COBYLA", 512),
    ("C07", 2, "SPSA", 256),
    ("C08", 2, "SPSA", 512),

    ("C09", 3, "COBYLA", 256),
    ("C10", 3, "COBYLA", 512),
    ("C11", 3, "SPSA", 256),
    ("C12", 3, "SPSA", 512),
]

CONFIG_IDS = {x[0] for x in CONFIGS}
RUN_SEEDS = (1000, 1001)
NOISY_CONDITIONS = ("N1", "N2")

# One graph is complete when it has:
# N0: 12 configs x 2 seeds = 24 rows
# N1/N2: 12 configs x 2 seeds x 2 noise conditions = 48 rows
EXPECTED_N0_ROWS_PER_GRAPH = len(CONFIGS) * len(RUN_SEEDS)
EXPECTED_NOISY_ROWS_PER_GRAPH = (
    len(CONFIGS) * len(RUN_SEEDS) * len(NOISY_CONDITIONS)
)


# ===========================================================================
# ARGUMENT PARSER
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the NQComp QAOA experiment in fast, resumable batches."
    )

    parser.add_argument(
        "--mode",
        choices=["n0", "noise"],
        required=True,
        help="Experiment mode: n0 or noise.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch number (1-indexed).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of graphs per batch.",
    )

    parser.add_argument(
        "--graph-ids",
        nargs="+",
        default=None,
        help=(
            "Run only the specified graph IDs. "
            "When supplied, --batch and --batch-size are ignored."
        ),
    )

    return parser.parse_args()


# ===========================================================================
# MANIFEST
# ===========================================================================

def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Graph manifest not found:\n{MANIFEST_PATH}\n"
            "Generate it first with: python -m src.graph_generation"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    required_columns = {
        "graph_id",
        "graph_family",
        "num_nodes",
        "graph_seed",
    }

    missing = required_columns - set(manifest.columns)

    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    return manifest


# ===========================================================================
# GRAPH CONSTRUCTION
# ===========================================================================

def build_graph(row: pd.Series):
    family = str(row["graph_family"])
    num_nodes = int(row["num_nodes"])
    graph_seed = int(row["graph_seed"])

    if family == "erdos_renyi":
        probability = float(row.get("probability", 0.35))

        spec = GraphSpec(
            graph_id=str(row["graph_id"]),
            family=family,
            num_nodes=num_nodes,
            seed=graph_seed,
            probability=probability,
            degree=None,
        )

    elif family == "random_regular":
        degree = int(row.get("degree", 3))

        spec = GraphSpec(
            graph_id=str(row["graph_id"]),
            family=family,
            num_nodes=num_nodes,
            seed=graph_seed,
            probability=None,
            degree=degree,
        )

    else:
        raise ValueError(f"Unsupported graph family: {family}")

    return generate_graph(spec)


# ===========================================================================
# RESULT LOADING / MERGING
# ===========================================================================

def load_existing_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)

        if df.empty:
            return pd.DataFrame()

        return df

    except Exception as exc:
        print(f"Warning: could not read existing results {path}: {exc}")
        return pd.DataFrame()


def merge_results(
    old_results: pd.DataFrame,
    new_results: pd.DataFrame,
) -> pd.DataFrame:
    if old_results.empty:
        combined = new_results.copy()
    elif new_results.empty:
        combined = old_results.copy()
    else:
        combined = pd.concat(
            [old_results, new_results],
            ignore_index=True,
        )

    if combined.empty:
        return combined

    key_columns = [
        "graph_id",
        "config_id",
        "noise_condition",
        "run_seed",
    ]

    available_keys = [
        column
        for column in key_columns
        if column in combined.columns
    ]

    if available_keys:
        combined = combined.drop_duplicates(
            subset=available_keys,
            keep="last",
        )

    sort_columns = [
        column
        for column in ["graph_id", "noise_condition", "config_id", "run_seed"]
        if column in combined.columns
    ]

    if sort_columns:
        combined = combined.sort_values(sort_columns)

    return combined.reset_index(drop=True)


def save_incremental(
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """
    Merge and immediately persist results.

    This is called after EVERY graph so completed work survives a
    later interruption if the CSV itself is persisted.
    """
    final_results = merge_results(existing, new_rows)

    save_raw_results(
        final_results,
        output_path,
    )

    return final_results


# ===========================================================================
# GRAPH SELECTION
# ===========================================================================

def select_graphs(
    manifest: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:

    if args.graph_ids:
        requested_ids = [str(graph_id) for graph_id in args.graph_ids]

        manifest_ids = set(manifest["graph_id"].astype(str))

        missing_requested = set(requested_ids) - manifest_ids

        if missing_requested:
            raise ValueError(
                "Requested graph IDs not found in manifest: "
                f"{sorted(missing_requested)}"
            )

        selected = manifest[
            manifest["graph_id"].astype(str).isin(requested_ids)
        ].copy()

        order = {
            graph_id: index
            for index, graph_id in enumerate(requested_ids)
        }

        selected["_requested_order"] = (
            selected["graph_id"]
            .astype(str)
            .map(order)
        )

        return (
            selected
            .sort_values("_requested_order")
            .drop(columns="_requested_order")
            .reset_index(drop=True)
        )

    if args.batch < 1:
        raise ValueError("--batch must be >= 1")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    experiment_manifest = manifest.head(
        N0_GRAPHS if args.mode == "n0" else NOISY_GRAPHS
    ).copy()

    start_index = (args.batch - 1) * args.batch_size
    end_index = start_index + args.batch_size

    if start_index >= len(experiment_manifest):
        total_batches = (
            len(experiment_manifest) + args.batch_size - 1
        ) // args.batch_size

        raise ValueError(
            f"Batch {args.batch} does not exist. "
            f"Only {total_batches} batches."
        )

    return experiment_manifest.iloc[start_index:end_index].copy()


# ===========================================================================
# COMPLETION CHECKS
# ===========================================================================

def graph_rows(
    results: pd.DataFrame,
    graph_id: str,
) -> pd.DataFrame:
    if results.empty or "graph_id" not in results.columns:
        return pd.DataFrame()

    return results[
        results["graph_id"].astype(str) == str(graph_id)
    ].copy()


def completed_n0_keys(
    results: pd.DataFrame,
    graph_id: str,
) -> set[tuple[str, int]]:
    rows = graph_rows(results, graph_id)

    if rows.empty:
        return set()

    if "run_status" in rows.columns:
        rows = rows[
            rows["run_status"].astype(str) == "success"
        ]

    if "config_id" not in rows.columns or "run_seed" not in rows.columns:
        return set()

    return {
        (str(row["config_id"]), int(row["run_seed"]))
        for _, row in rows.iterrows()
    }


def completed_noise_keys(
    results: pd.DataFrame,
    graph_id: str,
    noise_condition: str,
) -> set[tuple[str, int]]:
    rows = graph_rows(results, graph_id)

    if rows.empty:
        return set()

    rows = rows[
        rows["noise_condition"].astype(str) == noise_condition
    ]

    if "run_status" in rows.columns:
        rows = rows[
            rows["run_status"].astype(str) == "success"
        ]

    if "config_id" not in rows.columns or "run_seed" not in rows.columns:
        return set()

    return {
        (str(row["config_id"]), int(row["run_seed"]))
        for _, row in rows.iterrows()
    }


def is_n0_complete(
    results: pd.DataFrame,
    graph_id: str,
) -> bool:
    return len(completed_n0_keys(results, graph_id)) >= EXPECTED_N0_ROWS_PER_GRAPH


def is_noise_complete(
    results: pd.DataFrame,
    graph_id: str,
) -> bool:
    return all(
        len(completed_noise_keys(results, graph_id, noise))
        >= len(CONFIGS) * len(RUN_SEEDS)
        for noise in NOISY_CONDITIONS
    )


# ===========================================================================
# N0
# ===========================================================================

def run_n0_graph(
    graph: object,
    row: pd.Series,
    existing_results: pd.DataFrame,
) -> pd.DataFrame:

    graph_id = str(row["graph_id"])

    existing_keys = completed_n0_keys(
        existing_results,
        graph_id,
    )

    if is_n0_complete(existing_results, graph_id):
        print("  ✓ N0 already complete — skipping")
        return pd.DataFrame()

    print("  N0 — optimized QAOA")

    result = evaluate_graph(
        graph=graph,
        graph_id=graph_id,
        graph_family=str(row["graph_family"]),
        graph_seed=int(row["graph_seed"]),
        noise_condition="N0",
        run_seeds=RUN_SEEDS,
        exact_max_nodes=20,
        max_circuit_executions=80,
    )

    if not isinstance(result, pd.DataFrame):
        result = pd.DataFrame(result)

    # Do not add duplicate successful keys.
    if not result.empty:
        mask = []

        for _, r in result.iterrows():
            key = (
                str(r["config_id"]),
                int(r["run_seed"]),
            )
            mask.append(key not in existing_keys)

        result = result.loc[mask].copy()

    print(f"  N0 new rows: {len(result)}")

    return result


# ===========================================================================
# N1 / N2
# ===========================================================================

def run_noise_graph(
    graph: object,
    row: pd.Series,
    existing_noise_results: pd.DataFrame,
    n0_results: pd.DataFrame,
) -> pd.DataFrame:

    graph_id = str(row["graph_id"])

    if n0_results.empty:
        raise RuntimeError(
            f"No N0 results available for {graph_id}. "
            "N1/N2 cannot run without N0 optimal parameters."
        )

    # Restrict to this graph.
    n0_graph_results = graph_rows(
        n0_results,
        graph_id,
    )

    if n0_graph_results.empty:
        raise RuntimeError(
            f"No N0 rows found for {graph_id}."
        )

    # Only successful N0 rows can supply parameters.
    if "run_status" in n0_graph_results.columns:
        n0_graph_results = n0_graph_results[
            n0_graph_results["run_status"].astype(str) == "success"
        ].copy()

    new_rows: list[pd.DataFrame] = []

    for noise_condition in NOISY_CONDITIONS:

        existing_keys = completed_noise_keys(
            existing_noise_results,
            graph_id,
            noise_condition,
        )

        expected_keys = {
            (config_id, seed)
            for config_id in CONFIG_IDS
            for seed in RUN_SEEDS
        }

        missing_keys = expected_keys - existing_keys

        if not missing_keys:
            print(
                f"  ✓ {noise_condition} already complete — skipping"
            )
            continue

        # Only send the missing N0 rows into the noisy evaluator.
        missing_n0 = n0_graph_results[
            n0_graph_results.apply(
                lambda r: (
                    str(r["config_id"]),
                    int(r["run_seed"]),
                ) in missing_keys,
                axis=1,
            )
        ].copy()

        print(
            f"  {noise_condition} — "
            f"{len(missing_n0)} missing fixed-parameter evaluations"
        )

        result = evaluate_noisy_graph(
            graph=graph,
            graph_id=graph_id,
            graph_family=str(row["graph_family"]),
            graph_seed=int(row["graph_seed"]),
            noise_condition=noise_condition,
            n0_results=missing_n0,
            exact_max_nodes=20,
        )

        if not isinstance(result, pd.DataFrame):
            result = pd.DataFrame(result)

        new_rows.append(result)

        print(
            f"  ✓ {noise_condition} complete: "
            f"{len(result)} rows"
        )

    if not new_rows:
        return pd.DataFrame()

    return pd.concat(
        new_rows,
        ignore_index=True,
    )


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:

    args = parse_args()

    manifest = load_manifest()

    batch_manifest = select_graphs(
        manifest,
        args,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.mode == "n0":
        output_path = N0_OUTPUT
        noise_conditions = ("N0",)
        existing_results = load_existing_results(N0_OUTPUT)

    else:
        output_path = NOISY_OUTPUT
        noise_conditions = NOISY_CONDITIONS
        existing_results = load_existing_results(NOISY_OUTPUT)

    # N1/N2 always depend on the N0 CSV.
    n0_results = (
        load_existing_results(N0_OUTPUT)
        if args.mode == "noise"
        else pd.DataFrame()
    )

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------

    if args.graph_ids:
        mode_label = (
            "NOISE — EXPLICIT GRAPH RECOVERY"
            if args.mode == "noise"
            else "N0 — EXPLICIT GRAPH RECOVERY"
        )
    else:
        mode_label = (
            "N0"
            if args.mode == "n0"
            else "NOISE"
        )

    print()
    print("=" * 70)
    print("NQComp 2027 — FAST QAOA EXPERIMENT")
    print("=" * 70)
    print(f"Mode:                       {mode_label}")
    print(f"Total graphs:               {len(manifest)}")
    print(f"Graphs in this batch:       {len(batch_manifest)}")

    if args.graph_ids:
        print("Graph selection:            explicit IDs")
    else:
        print(f"Batch:                      {args.batch}")
        print(
            f"Batch range:                "
            f"{batch_manifest.iloc[0]['graph_id']} → "
            f"{batch_manifest.iloc[-1]['graph_id']}"
        )

    print(
        f"Noise conditions:           "
        f"{', '.join(noise_conditions)}"
    )

    if args.mode == "n0":
        evaluations_per_graph = EXPECTED_N0_ROWS_PER_GRAPH
    else:
        evaluations_per_graph = EXPECTED_N0_ROWS_PER_GRAPH * 2

    print(
        f"Maximum evaluations/graph: {evaluations_per_graph}"
    )
    print(f"Output:                     {output_path}")
    print("=" * 70)
    print()

    # -----------------------------------------------------------------------
    # Process graphs one at a time.
    # -----------------------------------------------------------------------

    for graph_position, (_, row) in enumerate(
        batch_manifest.iterrows(),
        start=1,
    ):

        graph_id = str(row["graph_id"])

        progress = (
            graph_position
            / len(batch_manifest)
            * 100.0
        )

        print(
            f"[Graph {graph_position}/{len(batch_manifest)}] "
            f"{graph_id} — {progress:.1f}%"
        )

        # Build graph.
        graph = build_graph(row)

        # ---------------------------------------------------------------
        # N0
        # ---------------------------------------------------------------

        if args.mode == "n0":

            new_rows = run_n0_graph(
                graph=graph,
                row=row,
                existing_results=existing_results,
            )

            # SAVE IMMEDIATELY AFTER THIS GRAPH.
            if not new_rows.empty:
                existing_results = save_incremental(
                    existing=existing_results,
                    new_rows=new_rows,
                    output_path=output_path,
                )
            else:
                # Re-read/retain existing CSV state.
                existing_results = load_existing_results(
                    output_path
                )

            current_rows = graph_rows(
                existing_results,
                graph_id,
            )

            print(
                f"  Saved checkpoint: "
                f"{len(current_rows)} rows for {graph_id}"
            )

            if is_n0_complete(
                existing_results,
                graph_id,
            ):
                print(f"  ✓ {graph_id} N0 COMPLETE")
            else:
                print(
                    f"  ⚠ {graph_id} N0 incomplete: "
                    f"{len(current_rows)}/"
                    f"{EXPECTED_N0_ROWS_PER_GRAPH} rows"
                )

        # ---------------------------------------------------------------
        # N1 / N2
        # ---------------------------------------------------------------

        else:

            # Refresh N0 in case it was generated by a previous command.
            n0_results = load_existing_results(N0_OUTPUT)

            n0_graph = graph_rows(
                n0_results,
                graph_id,
            )

            if len(completed_n0_keys(n0_results, graph_id)) < EXPECTED_N0_ROWS_PER_GRAPH:
                print(
                    f"  ⚠ N0 incomplete for {graph_id}: "
                    f"{len(completed_n0_keys(n0_results, graph_id))}/"
                    f"{EXPECTED_N0_ROWS_PER_GRAPH}"
                )
                print(
                    "  Skipping this graph. Complete N0 first."
                )
                continue

            new_rows = run_noise_graph(
                graph=graph,
                row=row,
                existing_noise_results=existing_results,
                n0_results=n0_graph,
            )

            # SAVE IMMEDIATELY AFTER THIS GRAPH.
            if not new_rows.empty:
                existing_results = save_incremental(
                    existing=existing_results,
                    new_rows=new_rows,
                    output_path=output_path,
                )
            else:
                existing_results = load_existing_results(
                    output_path
                )

            current_rows = graph_rows(
                existing_results,
                graph_id,
            )

            print(
                f"  Saved checkpoint: "
                f"{len(current_rows)} rows for {graph_id}"
            )

            if is_noise_complete(
                existing_results,
                graph_id,
            ):
                print(f"  ✓ {graph_id} N1/N2 COMPLETE")
            else:
                n1_count = len(
                    completed_noise_keys(
                        existing_results,
                        graph_id,
                        "N1",
                    )
                )
                n2_count = len(
                    completed_noise_keys(
                        existing_results,
                        graph_id,
                        "N2",
                    )
                )
                print(
                    f"  ⚠ {graph_id} incomplete: "
                    f"N1={n1_count}/24, "
                    f"N2={n2_count}/24"
                )

        print(
            f"  Overall progress: "
            f"{graph_position}/{len(batch_manifest)} "
            f"graphs ({progress:.2f}%)"
        )
        print()

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------

    final_results = load_existing_results(output_path)

    print()
    print("=" * 70)
    print("BATCH COMPLETE")
    print("=" * 70)
    print(f"Output:                    {output_path}")
    print(f"Total rows in output:      {len(final_results)}")

    if "graph_id" in final_results.columns:
        print(
            f"Unique graphs in output:   "
            f"{final_results['graph_id'].nunique()}"
        )

    if "noise_condition" in final_results.columns:
        print("Noise distribution:")
        print(
            final_results["noise_condition"].value_counts()
        )

    if "run_status" in final_results.columns:
        print("Run status:")
        print(
            final_results["run_status"].value_counts()
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
