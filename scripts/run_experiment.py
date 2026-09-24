"""Run the NQComp QAOA experiment in fast, resumable batches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.evaluation import (
    evaluate_graph,
    evaluate_noisy_graph,
    save_raw_results,
)
from src.graph_generation import (
    GraphSpec,
    build_manifest,
    generate_graph,
)


N0_GRAPHS = 100
NOISY_GRAPHS = 20
NOISE_CONDITIONS = ("N1", "N2")
RUN_SEED_START = 20000


def select_noisy_graphs(manifest: pd.DataFrame) -> pd.DataFrame:
    """Select 20 representative graphs for N1/N2."""

    selected = []

    families = manifest["family"].unique()

    for family in families:
        family_df = manifest[
            manifest["family"] == family
        ]

        for node_count in sorted(
            family_df["num_nodes"].unique()
        ):
            subset = family_df[
                family_df["num_nodes"] == node_count
            ]

            # Select two deterministic instances.
            selected.append(subset.iloc[:2])

    result = pd.concat(
        selected,
        ignore_index=True,
    )

    return result.head(NOISY_GRAPHS)


def parse_parameters(value) -> tuple[float, ...]:
    """Convert CSV representation of optimal parameters back to floats."""

    if isinstance(value, (tuple, list)):
        return tuple(float(x) for x in value)

    text = str(value).strip()

    text = text.strip("()[]")

    if not text:
        raise ValueError(
            "Empty optimal_parameters value."
        )

    return tuple(
        float(x.strip())
        for x in text.split(",")
        if x.strip()
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny pipeline validation.",
    )

    parser.add_argument(
        "--mode",
        choices=("n0", "noise"),
        default="n0",
        help="Run ideal N0 optimization or noisy evaluation.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch number, starting from 1.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of graphs per batch.",
    )

    args = parser.parse_args()

    if args.batch < 1:
        raise ValueError(
            "Batch number must be >= 1."
        )

    if args.batch_size < 1:
        raise ValueError(
            "Batch size must be >= 1."
        )

    # ---------------------------------------------------------
    # MANIFEST
    # ---------------------------------------------------------

    if args.smoke:

        manifest = build_manifest(
        node_counts=(10,),
        instances_per_setting=1,
        )

        manifest = manifest.iloc[:1]

        print("Running SMOKE TEST")

        # Smoke test only validates the fast N0 pipeline.
        noise_conditions = ("N0",)

        output_path = Path(
            "data/results/smoke_results.csv"
    )

    else:

        manifest = build_manifest()

        if args.mode == "n0":

            manifest = manifest.head(
                N0_GRAPHS
            )

            noise_conditions = ("N0",)

            output_path = Path(
                "data/results/n0_results.csv"
            )

        else:

            manifest = select_noisy_graphs(
                manifest
            )

            noise_conditions = NOISE_CONDITIONS

            output_path = Path(
                "data/results/noisy_results.csv"
            )

    total_graphs = len(manifest)

    # ---------------------------------------------------------
    # BATCH
    # ---------------------------------------------------------

    start_index = (
        (args.batch - 1)
        * args.batch_size
    )

    end_index = min(
        start_index + args.batch_size,
        total_graphs,
    )

    if start_index >= total_graphs:

        print(
            f"Batch {args.batch} does not exist. "
            f"Only {total_graphs} graphs."
        )

        return

    batch_manifest = manifest.iloc[
        start_index:end_index
    ]

    # ---------------------------------------------------------
    # PROGRESS INFORMATION
    # ---------------------------------------------------------

    configs = 12

    total_evaluations = (
        total_graphs
        * configs
        * len(noise_conditions)
    )

    batch_evaluations = (
        len(batch_manifest)
        * configs
        * len(noise_conditions)
    )

    print()
    print("=" * 70)
    print("NQComp 2027 — FAST QAOA EXPERIMENT")
    print("=" * 70)
    print(
        f"Mode:                       {args.mode.upper()}"
    )
    print(
        f"Total graphs:               {total_graphs}"
    )
    print(
        f"Graphs in this batch:       "
        f"{len(batch_manifest)}"
    )
    print(
        f"Batch:                      {args.batch}"
    )
    print(
        f"Batch range:                "
        f"G{start_index + 1:04d} → G{end_index:04d}"
    )
    print(
        f"Noise conditions:           "
        f"{', '.join(noise_conditions)}"
    )
    print(
        f"Evaluations this batch:     "
        f"{batch_evaluations}"
    )
    print(
        f"Output:                     "
        f"{output_path}"
    )
    print("=" * 70)

    all_results = []

    # ---------------------------------------------------------
    # GRAPH LOOP
    # ---------------------------------------------------------

    for graph_position, (_, row) in enumerate(
        batch_manifest.iterrows(),
        start=1,
    ):

        spec = GraphSpec(
            graph_id=row["graph_id"],
            family=row["family"],
            num_nodes=int(row["num_nodes"]),
            seed=int(row["seed"]),
            probability=row["probability"],
            degree=int(row["degree"]) if pd.notna(row["degree"]) else None,
        )

        graph = generate_graph(spec)

        print()
        print(
            f"[Graph {graph_position}/"
            f"{len(batch_manifest)}] "
            f"{spec.graph_id} "
            f"— "
            f"{graph_position / len(batch_manifest) * 100:.1f}%"
        )

        graph_results = []

        # -----------------------------------------------------
        # N0
        # -----------------------------------------------------

        if args.mode == "n0":

            print(
                "  N0 — optimizing 12 configurations..."
            )

            results = evaluate_graph(
                graph=graph,
                graph_id=spec.graph_id,
                graph_family=spec.family,
                graph_seed=spec.seed,
                noise_condition="N0",
                run_seed_start=RUN_SEED_START,
                exact_max_nodes=20,
                max_optimizer_evals=80,
                max_circuit_executions=80,
            )

            graph_results.append(results)

        # -----------------------------------------------------
        # N1/N2
        # -----------------------------------------------------

        else:

            n0_path = Path(
                "data/results/n0_results.csv"
            )

            if not n0_path.exists():

                raise FileNotFoundError(
                    "N0 results not found. "
                    "Run mode n0 first."
                )

            n0_all = pd.read_csv(
                n0_path
            )

            n0_results = n0_all[
                n0_all["graph_id"].astype(str)
                == str(spec.graph_id)
            ].copy()

            if n0_results.empty:

                raise RuntimeError(
                    f"No N0 results found for "
                    f"{spec.graph_id}"
                )

            # Restore parameter tuples.
            n0_results[
                "optimal_parameters"
            ] = n0_results[
                "optimal_parameters"
            ].apply(parse_parameters)

            for noise_condition in noise_conditions:

                print(
                    f"  {noise_condition} — "
                    f"12 fixed-parameter evaluations"
                )

                results = evaluate_noisy_graph(
                    graph=graph,
                    graph_id=spec.graph_id,
                    graph_family=spec.family,
                    graph_seed=spec.seed,
                    noise_condition=noise_condition,
                    n0_results=n0_results,
                    exact_max_nodes=20,
                )

                graph_results.append(results)

        # -----------------------------------------------------
        # SAVE GRAPH RESULTS
        # -----------------------------------------------------

        graph_df = pd.concat(
            graph_results,
            ignore_index=True,
        )

        all_results.append(
            graph_df
        )

        completed_graphs = (
            start_index + graph_position
        )

        percentage = (
            completed_graphs
            / total_graphs
            * 100
        )

        print(
            f"  ✓ {spec.graph_id} complete"
        )

        print(
            f"  Overall progress: "
            f"{completed_graphs}/"
            f"{total_graphs} graphs "
            f"({percentage:.2f}%)"
        )

    # ---------------------------------------------------------
    # COMBINE BATCH
    # ---------------------------------------------------------

    batch_results = pd.concat(
        all_results,
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------
    if args.smoke:
        # Smoke tests must always produce a fresh file.
        final_results = batch_results
    else:
        # Full experiments are resumable.
        if output_path.exists():
            existing = pd.read_csv(output_path)

            completed_graph_ids = set(
                batch_results["graph_id"].astype(str)
            )

            existing = existing[
                ~existing["graph_id"].astype(str).isin(
                    completed_graph_ids
                )
            ]

            final_results = pd.concat(
                [existing, batch_results],
                ignore_index=True,
            )

        else:
            final_results = batch_results

    final_results = final_results.sort_values(
    ["graph_id", "noise_condition", "config_id"]
    ).reset_index(drop=True)

    save_raw_results(
    final_results,
    output_path,
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    completed_graphs = (
        final_results[
            "graph_id"
        ].nunique()
    )

    completed_rows = len(
        final_results
    )

    expected_rows_per_graph = (
    configs * len(noise_conditions)
    )

    completed_evaluations = (
    completed_graphs * expected_rows_per_graph
    )

    overall_percentage = (
    completed_evaluations
    / total_evaluations
    * 100
    )
    print()
    print("=" * 70)
    print("BATCH COMPLETE")
    print("=" * 70)
    print(
        f"Graphs completed:          "
        f"{completed_graphs}/{total_graphs}"
    )
    print(
        f"Rows completed:            "
        f"{completed_evaluations}/{total_evaluations}"
    )
    print(
        f"Progress:                  "
        f"{overall_percentage:.2f}%"
    )
    print(
        f"Output:                    "
        f"{output_path}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()