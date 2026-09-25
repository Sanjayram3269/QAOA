"""Recover only missing N1/N2 noisy QAOA graph results.

This script:
- Loads the complete 100-graph manifest.
- Loads N0 results.
- Loads existing noisy results if available.
- Detects which graphs are incomplete.
- Runs ONLY missing graphs.
- Runs N1 and N2 using the fixed parameters from N0.
- Saves noisy_results.csv after EVERY completed graph.
- Is safe to restart/resume.
"""

from __future__ import annotations

from pathlib import Path
import sys

import networkx as nx
import pandas as pd

# Allow execution as:
# python scripts/recover_missing_noisy.py
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph_generation import GraphSpec, generate_graph
from src.evaluation import evaluate_noisy_graph


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

MANIFEST_PATH = ROOT / "data" / "graphs" / "master_graph_manifest.csv"
N0_PATH = ROOT / "data" / "results" / "n0_results.csv"
NOISY_PATH = ROOT / "data" / "results" / "noisy_results.csv"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

NOISE_CONDITIONS = ("N1", "N2")


def load_graph(row: pd.Series) -> nx.Graph:
    """Recreate a graph deterministically from the manifest row."""

    family = str(row["graph_family"])
    num_nodes = int(row["num_nodes"])
    seed = int(row["graph_seed"])

    probability = None
    degree = None

    if family == "erdos_renyi":
        probability = float(row["probability"])

    elif family == "random_regular":
        degree = int(row["degree"])

    else:
        raise ValueError(
            f"Unsupported graph family: {family}"
        )

    spec = GraphSpec(
        graph_id=str(row["graph_id"]),
        family=family,
        num_nodes=num_nodes,
        seed=seed,
        probability=probability,
        degree=degree,
    )

    return generate_graph(spec)


def is_graph_complete(
    noisy: pd.DataFrame,
    graph_id: str,
) -> bool:
    """Check whether a graph has complete N1 + N2 results."""

    if noisy.empty:
        return False

    graph_rows = noisy[
        noisy["graph_id"].astype(str) == str(graph_id)
    ]

    # 12 canonical configs × 1 run seed × 2 noise conditions = 24 rows.
    if len(graph_rows) != 24:
        return False

    for noise in NOISE_CONDITIONS:
        noise_rows = graph_rows[
            graph_rows["noise_condition"] == noise
        ]

        if len(noise_rows) != 12:
            return False

    return (
        graph_rows["run_status"]
        .astype(str)
        .eq("success")
        .all()
    )


def main() -> None:

    print("=" * 70)
    print("NQComp 2027 — MISSING NOISY GRAPH RECOVERY")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Validate required files
    # -----------------------------------------------------------------

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Graph manifest not found:\n{MANIFEST_PATH}"
        )

    if not N0_PATH.exists():
        raise FileNotFoundError(
            f"N0 results not found:\n{N0_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)
    n0_results = pd.read_csv(N0_PATH)

    print(f"Manifest: {manifest.shape}")
    print(f"N0 results: {n0_results.shape}")

    # -----------------------------------------------------------------
    # Load existing noisy results if available
    # -----------------------------------------------------------------

    if NOISY_PATH.exists():

        noisy_results = pd.read_csv(NOISY_PATH)

        print(
            f"Existing noisy results: "
            f"{noisy_results.shape}"
        )

    else:

        noisy_results = pd.DataFrame()

        print("No existing noisy_results.csv found.")
        print("Starting a fresh recovery file.")

    # -----------------------------------------------------------------
    # Determine missing graphs
    # -----------------------------------------------------------------

    missing_rows = []

    for _, row in manifest.iterrows():

        graph_id = str(row["graph_id"])

        if not is_graph_complete(
            noisy_results,
            graph_id,
        ):
            missing_rows.append(row)

    print()
    print(f"Total graphs: {len(manifest)}")
    print(
        f"Completed graphs: "
        f"{len(manifest) - len(missing_rows)}"
    )
    print(
        f"Missing/incomplete graphs: "
        f"{len(missing_rows)}"
    )

    if not missing_rows:

        print()
        print("ALL 100 GRAPHS ARE COMPLETE.")
        print("Nothing to recover.")
        return

    missing_ids = [
        str(row["graph_id"])
        for row in missing_rows
    ]

    print()
    print("Missing IDs:")
    print(missing_ids)
    print()

    # -----------------------------------------------------------------
    # Process missing graphs one by one
    # -----------------------------------------------------------------

    for position, row in enumerate(
        missing_rows,
        start=1,
    ):

        graph_id = str(row["graph_id"])

        print("=" * 70)
        print(
            f"[{position}/{len(missing_rows)}] "
            f"{graph_id} | "
            f"{row['graph_family']} | "
            f"n={int(row['num_nodes'])}"
        )
        print("=" * 70)

        # -------------------------------------------------------------
        # Recreate graph
        # -------------------------------------------------------------

        graph = load_graph(row)

        # -------------------------------------------------------------
        # IMPORTANT:
        # Filter N0 results to THIS graph only.
        #
        # This prevents passing all 100 graphs' N0 rows into
        # evaluate_noisy_graph().
        # -------------------------------------------------------------

        graph_n0 = n0_results[
            n0_results["graph_id"].astype(str)
            == graph_id
        ].copy()

        if graph_n0.empty:
            raise ValueError(
                f"No N0 results found for {graph_id}"
            )

        if len(graph_n0) != 12:
            raise ValueError(
                f"Expected 12 N0 configuration rows for "
                f"{graph_id}, found {len(graph_n0)}"
            )

        # -------------------------------------------------------------
        # Remove any incomplete/old rows for this graph before adding
        # the freshly recovered results.
        # -------------------------------------------------------------

        if not noisy_results.empty:

            noisy_results = noisy_results[
                noisy_results["graph_id"].astype(str)
                != graph_id
            ].copy()

        graph_new_results = []

        # -------------------------------------------------------------
        # N1 + N2
        # -------------------------------------------------------------

        for noise_condition in NOISE_CONDITIONS:

            print(
                f"  {noise_condition} — "
                f"running 12 fixed-parameter evaluations..."
            )

            result = evaluate_noisy_graph(
                graph=graph,
                graph_id=graph_id,
                graph_family=str(row["graph_family"]),
                graph_seed=int(row["graph_seed"]),
                noise_condition=noise_condition,
                n0_results=graph_n0,
                exact_max_nodes=20,
            )

            if not isinstance(result, pd.DataFrame):
                result = pd.DataFrame(result)

            if len(result) != 12:
                raise ValueError(
                    f"{graph_id} {noise_condition}: "
                    f"expected 12 rows, got {len(result)}"
                )

            graph_new_results.append(result)

            print(
                f"  ✓ {noise_condition} complete: "
                f"{len(result)} rows"
            )

        # -------------------------------------------------------------
        # Combine N1 + N2
        # -------------------------------------------------------------

        graph_results = pd.concat(
            graph_new_results,
            ignore_index=True,
        )

        if len(graph_results) != 24:
            raise ValueError(
                f"{graph_id}: expected 24 noisy rows, "
                f"got {len(graph_results)}"
            )

        # -------------------------------------------------------------
        # Append to cumulative result
        # -------------------------------------------------------------

        if noisy_results.empty:
            noisy_results = graph_results.copy()

        else:
            noisy_results = pd.concat(
                [
                    noisy_results,
                    graph_results,
                ],
                ignore_index=True,
            )

        # -------------------------------------------------------------
        # Sort for stable deterministic output
        # -------------------------------------------------------------

        sort_columns = [
            "graph_id",
            "noise_condition",
            "config_id",
            "run_seed",
        ]

        existing_sort_columns = [
            col
            for col in sort_columns
            if col in noisy_results.columns
        ]

        noisy_results = noisy_results.sort_values(
            existing_sort_columns
        ).reset_index(drop=True)

        # -------------------------------------------------------------
        # CRITICAL:
        # SAVE IMMEDIATELY AFTER EVERY GRAPH.
        # -------------------------------------------------------------

        NOISY_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        noisy_results.to_csv(
            NOISY_PATH,
            index=False,
        )

        print(
            f"  ✓ SAVED: {NOISY_PATH}"
        )

        print(
            f"  Current noisy results: "
            f"{noisy_results.shape}"
        )

    # -----------------------------------------------------------------
    # Final verification
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("RECOVERY COMPLETE")
    print("=" * 70)

    final = pd.read_csv(NOISY_PATH)

    print(f"Final noisy results: {final.shape}")

    graph_counts = (
        final.groupby("graph_id")
        .size()
    )

    complete_graphs = (
        graph_counts[graph_counts == 24]
        .index
        .tolist()
    )

    incomplete_graphs = [
        graph_id
        for graph_id in manifest["graph_id"].astype(str)
        if graph_id not in complete_graphs
    ]

    print(
        f"Complete graphs: "
        f"{len(complete_graphs)}"
    )

    print(
        f"Incomplete graphs: "
        f"{len(incomplete_graphs)}"
    )

    if incomplete_graphs:

        print(
            "Still incomplete:",
            incomplete_graphs,
        )

    else:

        print()
        print("SUCCESS: ALL 100 GRAPHS COMPLETE.")
        print("Expected noisy rows: 2400")
        print(f"Actual noisy rows: {len(final)}")


if __name__ == "__main__":
    main()