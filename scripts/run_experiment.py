"""Run reproducible NQComp quantum experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation import evaluate_graph, save_raw_results
from src.graph_generation import build_manifest, generate_graph


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
            node_counts=(10,),
            instances_per_setting=1,
        )
        manifest = manifest.iloc[:1]
        noise_conditions = ("N0")
        output_path = Path("data/results/smoke_results.csv")
    else:
        manifest = build_manifest()
        noise_conditions = ("N0", "N1", "N2")
        output_path = Path("data/results/raw_results.csv")

    all_results = []

    for _, row in manifest.iterrows():
        graph_spec = row.to_dict()

        from src.graph_generation import GraphSpec

        spec = GraphSpec(
            graph_id=graph_spec["graph_id"],
            family=graph_spec["family"],
            num_nodes=int(graph_spec["num_nodes"]),
            seed=int(graph_spec["seed"]),
            probability=graph_spec["probability"],
            degree=graph_spec["degree"],
        )

        graph = generate_graph(spec)

        for noise_condition in noise_conditions:
            results = evaluate_graph(
                graph=graph,
                graph_id=spec.graph_id,
                graph_family=spec.family,
                graph_seed=spec.seed,
                noise_condition=noise_condition,
                run_seed_start=20000,
                exact_max_nodes=20,
                max_optimizer_evals=80,
            )
            all_results.append(results)

    import pandas as pd

    combined = pd.concat(all_results, ignore_index=True)
    save_raw_results(combined, output_path)

    print(f"Wrote {len(combined)} rows to {output_path}")


if __name__ == "__main__":
    main()