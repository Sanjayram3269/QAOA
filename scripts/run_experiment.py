"""Run reproducible NQComp quantum experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation import evaluate_graph, save_raw_results
from src.graph_generation import GraphSpec, build_manifest, generate_graph


def _optional_float(value) -> float | None:
    return None if pd.isna(value) else float(value)


def _optional_int(value) -> int | None:
    return None if pd.isna(value) else int(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny experiment for pipeline validation.",
    )
    args = parser.parse_args()

    if args.smoke:
        manifest = build_manifest(
            families=("erdos_renyi",),
            node_counts=(10,),
            instances_per_setting=1,
        )
        noise_conditions = ("N0",)
        run_seeds = (20000,)
        max_circuit_executions = 10
        output_path = Path(
            "data/raw/batch_sanjay/smoke_results.csv"
        )
    else:
        manifest = build_manifest(
            instances_per_setting=5,
            random_regular_degree=4,
        )
        noise_conditions = ("N0", "N1", "N2")
        run_seeds = (20000, 20001, 20002)
        max_circuit_executions = 200
        output_path = Path(
            "data/raw/batch_sanjay/feasibility_results.csv"
        )

    all_results = []

    for _, row in manifest.iterrows():
        spec = GraphSpec(
            graph_id=str(row["graph_id"]),
            family=str(row["family"]),
            num_nodes=int(row["num_nodes"]),
            seed=int(row["seed"]),
            probability=_optional_float(row["probability"]),
            degree=_optional_int(row["degree"]),
        )
        graph = generate_graph(spec)

        for noise_condition in noise_conditions:
            results = evaluate_graph(
                graph=graph,
                graph_id=spec.graph_id,
                graph_family=spec.family,
                graph_seed=spec.seed,
                noise_condition=noise_condition,
                run_seeds=run_seeds,
                exact_max_nodes=20,
                max_circuit_executions=max_circuit_executions,
            )
            all_results.append(results)

    combined = pd.concat(all_results, ignore_index=True)
    save_raw_results(combined, output_path)

    failed = int((combined["run_status"] != "success").sum())
    print(f"Wrote {len(combined)} rows to {output_path}")
    print(f"Successful rows: {len(combined) - failed}; failed rows: {failed}")


if __name__ == "__main__":
    main()
