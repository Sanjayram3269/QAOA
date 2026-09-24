"""
Run the NQComp QAOA experiment in fast, resumable batches.

Supports:
- N0 optimized QAOA runs
- N1/N2 fixed-parameter noisy evaluations
- positional batch execution
- explicit graph-ID recovery using --graph-ids
- resumable CSV output
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
from src.graph_generation import generate_graph
from src.graph_generation import GraphSpec


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

MANIFEST_PATH = (
    ROOT / "data" / "graphs" / "master_graph_manifest.csv"
)

# ---------------------------------------------------------------------------
# The 12 fixed configurations used by the NQComp experiment.
# ---------------------------------------------------------------------------

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


# ===========================================================================
# ARGUMENT PARSER
# ===========================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run the NQComp QAOA experiment in "
            "fast, resumable batches."
        )
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

    # -----------------------------------------------------------------------
    # NEW:
    # Explicit graph-ID recovery mode.
    #
    # Example:
    #
    # --graph-ids G0003 G0004 G0005
    #
    # When supplied, positional batching is ignored.
    # -----------------------------------------------------------------------

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
            "Generate master_graph_manifest.csv first."
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
            "Manifest is missing required columns: "
            f"{sorted(missing)}"
        )

    return manifest


# ===========================================================================
# GRAPH CONSTRUCTION
# ===========================================================================

def build_graph(row: pd.Series):

    family = row["graph_family"]
    num_nodes = int(row["num_nodes"])
    graph_seed = int(row["graph_seed"])

    if family == "erdos_renyi":

        probability = float(
            row.get("probability", 0.35)
        )

        spec = GraphSpec(
            graph_id=str(row["graph_id"]),
            family=family,
            num_nodes=num_nodes,
            seed=graph_seed,
            probability=probability,
            degree=None,
        )

    elif family == "random_regular":

        degree = int(
            row.get("degree", 3)
        )

        spec = GraphSpec(
            graph_id=str(row["graph_id"]),
            family=family,
            num_nodes=num_nodes,
            seed=graph_seed,
            probability=None,
            degree=degree,
        )

    else:
        raise ValueError(
            f"Unsupported graph family: {family}"
        )

    return generate_graph(spec)


# ===========================================================================
# EXISTING RESULTS / RESUMABILITY
# ===========================================================================

def load_existing_results(path: Path) -> pd.DataFrame | None:

    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)

        if df.empty:
            return None

        return df

    except Exception as exc:
        print(
            f"Warning: could not read existing results "
            f"{path}: {exc}"
        )

        return None


def merge_results(
    old_results: pd.DataFrame | None,
    new_results: pd.DataFrame,
) -> pd.DataFrame:

    if old_results is None or old_results.empty:
        return new_results.copy()

    if new_results.empty:
        return old_results.copy()

    combined = pd.concat(
        [old_results, new_results],
        ignore_index=True,
    )

    # -----------------------------------------------------------------------
    # Prevent duplicate experiment rows.
    #
    # A row is uniquely identified by graph/config/noise/run seed.
    # -----------------------------------------------------------------------

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

    return combined.reset_index(drop=True)


# ===========================================================================
# GRAPH SELECTION
# ===========================================================================

def select_graphs(
    manifest: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:

    # -----------------------------------------------------------------------
    # Explicit graph-ID mode.
    #
    # This is used for recovering the missing 16 graphs:
    #
    # G0003 ... G0010
    # G0013 ... G0020
    # -----------------------------------------------------------------------

    if args.graph_ids:

        requested_ids = set(
            str(graph_id)
            for graph_id in args.graph_ids
        )

        manifest_ids = set(
            manifest["graph_id"].astype(str)
        )

        missing_requested = (
            requested_ids - manifest_ids
        )

        if missing_requested:
            raise ValueError(
                "Requested graph IDs not found in manifest: "
                f"{sorted(missing_requested)}"
            )

        batch_manifest = manifest[
            manifest["graph_id"].astype(str).isin(
                requested_ids
            )
        ].copy()

        # Preserve the authoritative manifest ordering.
        batch_manifest["_requested_order"] = (
            batch_manifest["graph_id"]
            .astype(str)
            .map(
                {
                    graph_id: index
                    for index, graph_id
                    in enumerate(args.graph_ids)
                }
            )
        )

        batch_manifest = (
            batch_manifest
            .sort_values("_requested_order")
            .drop(columns="_requested_order")
            .reset_index(drop=True)
        )

        return batch_manifest

    # -----------------------------------------------------------------------
    # Existing positional batch mode.
    # -----------------------------------------------------------------------

    if args.batch < 1:
        raise ValueError(
            "--batch must be >= 1"
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    if args.mode == "n0":
        experiment_manifest = manifest.head(
            N0_GRAPHS
        ).copy()

    else:
        experiment_manifest = manifest.head(
            NOISY_GRAPHS
        ).copy()

    start_index = (
        (args.batch - 1)
        * args.batch_size
    )

    end_index = (
        start_index
        + args.batch_size
    )

    if start_index >= len(experiment_manifest):

        total_batches = (
            len(experiment_manifest)
            + args.batch_size
            - 1
        ) // args.batch_size

        raise ValueError(
            f"Batch {args.batch} does not exist. "
            f"Only {total_batches} batches."
        )

    batch_manifest = experiment_manifest.iloc[
        start_index:end_index
    ].copy()

    return batch_manifest


# ===========================================================================
# MAIN EXPERIMENT
# ===========================================================================

def main():

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

    # -----------------------------------------------------------------------
    # Output file.
    # -----------------------------------------------------------------------

    if args.mode == "n0":
        output_path = N0_OUTPUT
        noise_conditions = ["N0"]

    else:
        output_path = NOISY_OUTPUT
        noise_conditions = ["N1", "N2"]

    # -----------------------------------------------------------------------
    # Display experiment information.
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

    print(
        f"Mode:                       {mode_label}"
    )

    print(
        f"Total graphs:               "
        f"{len(manifest)}"
    )

    print(
        f"Graphs in this batch:       "
        f"{len(batch_manifest)}"
    )

    if args.graph_ids:

        print(
            "Graph selection:            explicit IDs"
        )

    else:

        print(
            f"Batch:                      "
            f"{args.batch}"
        )

        print(
            f"Batch range:                "
            f"{batch_manifest.iloc[0]['graph_id']} "
            f"→ "
            f"{batch_manifest.iloc[-1]['graph_id']}"
        )

    print(
        f"Noise conditions:           "
        f"{', '.join(noise_conditions)}"
    )

    evaluations_per_graph = (
        len(CONFIGS)
        * len(noise_conditions)
    )

    print(
        f"Evaluations this batch:     "
        f"{len(batch_manifest) * evaluations_per_graph}"
    )

    print(
        f"Output:                     "
        f"{output_path}"
    )

    print("=" * 70)
    print()

    # -----------------------------------------------------------------------
    # Run each graph.
    # -----------------------------------------------------------------------

    all_new_results: list[pd.DataFrame] = []

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
            f"[Graph {graph_position}/"
            f"{len(batch_manifest)}] "
            f"{graph_id} — "
            f"{progress:.1f}%"
        )

        graph = build_graph(row)

        # -------------------------------------------------------------------
        # N0: optimized QAOA.
        # -------------------------------------------------------------------

        if args.mode == "n0":

            print(
                "  N0 — optimized QAOA"
            )

            result = evaluate_graph(
                graph=graph,
                graph_id=graph_id,
                graph_family=str(
                    row["graph_family"]
                ),
                graph_seed=int(
                    row["graph_seed"]
                ),
                noise_condition="N0",
                run_seeds=(1000, 1001),
                exact_max_nodes=20,
                max_circuit_executions=80,
            )

            if isinstance(result, pd.DataFrame):
                graph_results = result
            else:
                graph_results = pd.DataFrame(result)

            all_new_results.append(
                graph_results
            )

        # -------------------------------------------------------------------
        # N1/N2: fixed-parameter noisy evaluation.
        # -------------------------------------------------------------------

        else:

            for noise_condition in noise_conditions:

                print(
                    f"  {noise_condition} — "
                    f"{len(CONFIGS)} "
                    f"fixed-parameter evaluations"
                )

                result = evaluate_noisy_graph(
                    graph=graph,
                    graph_id=graph_id,
                    graph_family=str(
                        row["graph_family"]
                    ),
                    graph_seed=int(
                        row["graph_seed"]
                    ),
                    noise_condition=noise_condition,
                    configs=CONFIGS,
                    run_seeds=(1000, 1001),
                    exact_max_nodes=20,
                )

                if isinstance(result, pd.DataFrame):
                    graph_results = result
                else:
                    graph_results = pd.DataFrame(
                        result
                    )

                all_new_results.append(
                    graph_results
                )

        print(
            f"  ✓ {graph_id} complete"
        )

        print(
            f"  Overall progress: "
            f"{graph_position}/"
            f"{len(batch_manifest)} graphs "
            f"({progress:.2f}%)"
        )

    # =========================================================================
    # SAVE
    # =========================================================================

    if all_new_results:

        batch_results = pd.concat(
            all_new_results,
            ignore_index=True,
        )

    else:

        batch_results = pd.DataFrame()

    existing_results = load_existing_results(
        output_path
    )

    final_results = merge_results(
        existing_results,
        batch_results,
    )

    save_raw_results(
        final_results,
        output_path,
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print()
    print("=" * 70)
    print("BATCH COMPLETE")
    print("=" * 70)

    print(
        f"Graphs completed:          "
        f"{len(batch_manifest)}"
    )

    print(
        f"Rows in this batch:        "
        f"{len(batch_results)}"
    )

    print(
        f"Total rows in output:      "
        f"{len(final_results)}"
    )

    if "graph_id" in final_results.columns:

        print(
            f"Unique graphs in output:   "
            f"{final_results['graph_id'].nunique()}"
        )

    if "noise_condition" in final_results.columns:

        print(
            "Noise distribution:"
        )

        print(
            final_results[
                "noise_condition"
            ].value_counts()
        )

    if "run_status" in final_results.columns:

        print(
            "Run status:"
        )

        print(
            final_results[
                "run_status"
            ].value_counts()
        )

    print(
        f"Output:                    "
        f"{output_path}"
    )

    print("=" * 70)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    main()