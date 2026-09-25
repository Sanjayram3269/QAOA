import numpy as np
import pandas as pd

from src.features import build_features_from_raw_results
from src.graph_generation import GraphSpec, generate_graph
from src.ml_selector import (
    build_candidate_table,
    condition_global_best_predictions,
    evaluate_predictions,
    predict_feasible,
)
from src.splitting import attach_graph_splits, split_graphs


def _expanded_graph_dataset() -> pd.DataFrame:
    rows = []
    graph_number = 1
    for family in ("erdos_renyi", "random_regular"):
        for num_nodes in (10, 12, 15, 18, 20):
            for _ in range(5):
                for noise in ("N0", "N1", "N2"):
                    for budget in ("B256", "B512"):
                        rows.append(
                            {
                                "graph_id": f"G{graph_number:04d}",
                                "graph_family": family,
                                "num_nodes": num_nodes,
                                "noise_condition": noise,
                                "resource_budget": budget,
                            }
                        )
                graph_number += 1
    return pd.DataFrame(rows)


def test_graph_split_is_exact_reproducible_and_leakage_safe():
    dataset = _expanded_graph_dataset()
    first = split_graphs(dataset, seed=2027)
    second = split_graphs(dataset, seed=2027)

    pd.testing.assert_frame_equal(first, second)
    assert first["split"].value_counts().to_dict() == {
        "train": 35,
        "validation": 8,
        "test": 7,
    }

    expanded = attach_graph_splits(dataset, first)
    assert expanded.groupby("graph_id")["split"].nunique().max() == 1
    assert len(expanded) == 300


class _DummyProbabilityModel:
    classes_ = np.array(["C01", "C02"])

    def predict_proba(self, features):
        return np.tile(np.array([[0.1, 0.9]]), (len(features), 1))


def test_prediction_masks_configs_that_exceed_budget():
    rows = pd.DataFrame(
        [
            {
                "graph_id": "G1",
                "graph_family": "erdos_renyi",
                "num_nodes": 10,
                "num_edges": 15,
                "density": 1 / 3,
                "average_degree": 3.0,
                "degree_std": 1.0,
                "min_degree": 1,
                "max_degree": 5,
                "triangle_count": 2,
                "noise_condition": "N0",
                "resource_budget": "B256",
            },
            {
                "graph_id": "G1",
                "graph_family": "erdos_renyi",
                "num_nodes": 10,
                "num_edges": 15,
                "density": 1 / 3,
                "average_degree": 3.0,
                "degree_std": 1.0,
                "min_degree": 1,
                "max_degree": 5,
                "triangle_count": 2,
                "noise_condition": "N0",
                "resource_budget": "B512",
            },
        ]
    )
    predictions = predict_feasible(_DummyProbabilityModel(), rows)
    assert predictions["predicted_config_id"].tolist() == ["C01", "C02"]


def test_feature_reconstruction_uses_only_preselection_information():
    spec = GraphSpec(
        graph_id="G0001",
        family="erdos_renyi",
        num_nodes=10,
        seed=10000,
        probability=0.35,
    )
    graph = generate_graph(spec)
    raw = pd.DataFrame(
        [
            {
                "graph_id": spec.graph_id,
                "graph_family": spec.family,
                "num_nodes": spec.num_nodes,
                "num_edges": graph.number_of_edges(),
                "graph_seed": spec.seed,
                "noise_condition": "N0",
            }
        ]
    )
    features = build_features_from_raw_results(raw)
    assert len(features) == 2
    assert set(features["resource_budget"]) == {"B256", "B512"}
    assert "expected_approximation_ratio" not in features.columns
    assert (features["num_edges"] == graph.number_of_edges()).all()


def test_condition_global_best_is_fit_from_development_graphs_only():
    aggregated = pd.DataFrame(
        [
            {
                "graph_id": graph_id,
                "noise_condition": "N0",
                "config_id": config_id,
                "depth": 1,
                "shots_per_circuit": shots,
                "mean_expected_approximation_ratio": quality,
                "mean_total_executed_shots": cost,
            }
            for graph_id, config_id, shots, quality, cost in [
                ("G1", "C01", 256, 0.80, 25600),
                ("G1", "C02", 512, 0.90, 51200),
                ("G2", "C01", 256, 0.82, 25600),
                ("G2", "C02", 512, 0.88, 51200),
            ]
        ]
    )
    candidates = build_candidate_table(aggregated)
    evaluation = pd.DataFrame(
        [
            {
                "graph_id": "G2",
                "noise_condition": "N0",
                "resource_budget": budget,
            }
            for budget in ("B256", "B512")
        ]
    )
    predictions = condition_global_best_predictions(candidates, {"G1"}, evaluation)
    assert predictions.loc[
        predictions["resource_budget"] == "B256", "predicted_config_id"
    ].item() == "C01"

    labels = evaluation.assign(
        oracle_config_id=["C01", "C02"],
        oracle_utility=[0.59, 0.60],
        oracle_quality=[0.82, 0.88],
        oracle_cost=[25600, 51200],
    )
    metrics, scored = evaluate_predictions(predictions, labels, candidates)
    assert metrics["examples"] == 2
    assert len(scored) == 2
