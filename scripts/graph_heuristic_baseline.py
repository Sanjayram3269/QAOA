from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


INPUT = Path("data/ml/splits/ml_performance_common.csv")
OUTPUT = Path("data/ml/upgrades/graph_heuristic")

TARGET = "mean_expected_approximation_ratio"

GRAPH_FEATURES = ["num_nodes", "num_edges", "density"]

BUDGETS = {
    "B256": 256,
    "B512": 512,
}

K = 5


def load_data() -> pd.DataFrame:
    data = pd.read_csv(INPUT)

    required = [
        "graph_id",
        "config_id",
        "noise_condition",
        "depth",
        "optimizer",
        "shots",
        TARGET,
        "num_nodes",
        "num_edges",
        "density",
        "graph_family",
        "split",
    ]

    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return data


def build_graph_table(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data[
            ["graph_id", "graph_family"] + GRAPH_FEATURES + ["split"]
        ]
        .drop_duplicates("graph_id")
        .reset_index(drop=True)
    )


def nearest_training_graphs(
    target_graph: pd.Series,
    train_graphs: pd.DataFrame,
    scaler: StandardScaler,
    k: int,
) -> pd.DataFrame:
    X_train = scaler.transform(train_graphs[GRAPH_FEATURES])
    X_target = scaler.transform(
        target_graph[GRAPH_FEATURES].to_frame().T
    )

    n_neighbors = min(k, len(train_graphs))

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X_train)

    distances, indices = nn.kneighbors(X_target)

    result = train_graphs.iloc[indices[0]].copy()
    result["distance"] = distances[0]

    return result


def choose_configuration(
    data: pd.DataFrame,
    neighbours: pd.DataFrame,
    noise: str,
    budget_limit: int,
) -> tuple[str, float]:

    neighbour_ids = set(neighbours["graph_id"])

    candidates = data[
        data["graph_id"].isin(neighbour_ids)
        & data["noise_condition"].eq(noise)
        & (data["shots"] <= budget_limit)
    ].copy()

    if candidates.empty:
        raise ValueError(
            f"No feasible candidates for noise={noise}, "
            f"budget={budget_limit}"
        )

    # Average performance across the nearest training graphs.
    grouped = (
        candidates.groupby("config_id", as_index=False)[TARGET]
        .mean()
        .sort_values(
            [TARGET, "config_id"],
            ascending=[False, True],
        )
    )

    best = grouped.iloc[0]

    return str(best["config_id"]), float(best[TARGET])


def evaluate_split(
    data: pd.DataFrame,
    graph_table: pd.DataFrame,
    target_split: str,
    train_graphs: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:

    rows = []

    target_graphs = graph_table[
        graph_table["split"].eq(target_split)
    ]

    for _, graph in target_graphs.iterrows():

        neighbours = nearest_training_graphs(
            graph,
            train_graphs,
            scaler,
            K,
        )

        for noise in sorted(data["noise_condition"].unique()):

            for budget_name, budget_limit in BUDGETS.items():

                config_id, neighbour_quality = choose_configuration(
                    data,
                    neighbours,
                    noise,
                    budget_limit,
                )

                actual = data[
                    data["graph_id"].eq(graph["graph_id"])
                    & data["noise_condition"].eq(noise)
                    & data["config_id"].eq(config_id)
                    & (data["shots"] <= budget_limit)
                ]

                if actual.empty:
                    raise ValueError(
                        f"Missing selected configuration: "
                        f"{graph['graph_id']} {noise} "
                        f"{budget_name} {config_id}"
                    )

                actual = actual.iloc[0]

                feasible = data[
                    data["graph_id"].eq(graph["graph_id"])
                    & data["noise_condition"].eq(noise)
                    & (data["shots"] <= budget_limit)
                ]

                oracle = feasible[TARGET].max()

                rows.append(
                    {
                        "graph_id": graph["graph_id"],
                        "graph_family": graph["graph_family"],
                        "num_nodes": graph["num_nodes"],
                        "num_edges": graph["num_edges"],
                        "density": graph["density"],
                        "noise_condition": noise,
                        "budget": budget_name,
                        "selected_config_id": config_id,
                        "selected_quality": float(actual[TARGET]),
                        "neighbour_predicted_quality": neighbour_quality,
                        "oracle_quality": float(oracle),
                        "regret": max(
                            0.0,
                            float(oracle) - float(actual[TARGET]),
                        ),
                        "selected_is_oracle": bool(
                            np.isclose(
                                float(actual[TARGET]),
                                float(oracle),
                            )
                        ),
                        "neighbour_graph_ids": "|".join(
                            neighbours["graph_id"].astype(str)
                        ),
                    }
                )

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:

    scopes = [
        ("ALL", pd.Series(True, index=results.index)),
    ]

    scopes.extend(
        (name, results["budget"].eq(name))
        for name in BUDGETS
    )

    scopes.extend(
        (noise, results["noise_condition"].eq(noise))
        for noise in sorted(results["noise_condition"].unique())
    )

    rows = []

    for name, mask in scopes:

        part = results.loc[mask]

        rows.append(
            {
                "scope": name,
                "graphs": int(part["graph_id"].nunique()),
                "cases": int(len(part)),
                "heuristic_mean": float(
                    part["selected_quality"].mean()
                ),
                "oracle_mean": float(
                    part["oracle_quality"].mean()
                ),
                "mean_regret": float(
                    part["regret"].mean()
                ),
                "oracle_selection_accuracy": float(
                    part["selected_is_oracle"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:

    OUTPUT.mkdir(parents=True, exist_ok=True)

    data = load_data()

    graph_table = build_graph_table(data)

    train_graphs = graph_table[
        graph_table["split"].eq("train")
    ].copy()

    scaler = StandardScaler()
    scaler.fit(train_graphs[GRAPH_FEATURES])

    validation = evaluate_split(
        data,
        graph_table,
        "validation",
        train_graphs,
        scaler,
    )

    test = evaluate_split(
        data,
        graph_table,
        "test",
        train_graphs,
        scaler,
    )

    validation.to_csv(
        OUTPUT / "validation_results.csv",
        index=False,
    )

    test.to_csv(
        OUTPUT / "test_selected_configurations.csv",
        index=False,
    )

    validation_summary = summarize(validation)
    test_summary = summarize(test)

    validation_summary.to_csv(
        OUTPUT / "validation_summary.csv",
        index=False,
    )

    test_summary.to_csv(
        OUTPUT / "test_summary.csv",
        index=False,
    )

    manifest = {
        "method": "training-only graph nearest-neighbour heuristic",
        "input": str(INPUT),
        "target": TARGET,
        "graph_features": GRAPH_FEATURES,
        "k_neighbours": K,
        "distance": "standardized Euclidean distance",
        "training_graphs": int(len(train_graphs)),
        "validation_graphs": int(
            graph_table["split"].eq("validation").sum()
        ),
        "test_graphs": int(
            graph_table["split"].eq("test").sum()
        ),
        "budgets": BUDGETS,
        "selection_rule": (
            "Among feasible configurations, select the configuration "
            "with highest mean target performance across the K nearest "
            "training graphs for the same noise condition."
        ),
        "test_data_used_for_selection": False,
    }

    with open(
        OUTPUT / "heuristic_definition.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=2)

    print("\nValidation summary:")
    print(validation_summary.to_string(index=False))

    print("\nTest summary:")
    print(test_summary.to_string(index=False))

    print(f"\nOutputs written to {OUTPUT}")


if __name__ == "__main__":
    main()
