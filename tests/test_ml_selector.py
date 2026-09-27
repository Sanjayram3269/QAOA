from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml_selector import (
    TARGET,
    build_enhanced_graph_features,
    exact_sign_flip_pvalue,
    fixed_baseline_choices,
    select_configurations,
    validate_split_integrity,
)


class ConfigScoreModel:
    """Small deterministic model used to exercise budget selection."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features["score"].to_numpy()


class ScoreFeatureSet:
    columns = ["score"]


def complete_rows() -> pd.DataFrame:
    rows = []
    for split, graph in zip(("train", "validation", "test"), ("G1", "G2", "G3")):
        for config in range(1, 13):
            rows.append(
                {
                    "graph_id": graph,
                    "config_id": f"C{config:02d}",
                    "noise_condition": "N0",
                    "split": split,
                    "shots": 256 if config <= 6 else 512,
                    TARGET: config / 12,
                }
            )
    return pd.DataFrame(rows)


def test_split_validation_accepts_complete_graph_disjoint_data() -> None:
    validate_split_integrity(complete_rows())


def test_split_validation_rejects_graph_leakage() -> None:
    data = complete_rows()
    data.loc[data.index[-1], "graph_id"] = "G1"
    with pytest.raises(ValueError, match="Graph leakage"):
        validate_split_integrity(data)


def test_selection_respects_shots_per_circuit_budget() -> None:
    data = pd.DataFrame(
        {
            "graph_id": ["G1", "G1"],
            "noise_condition": ["N0", "N0"],
            "config_id": ["C01", "C02"],
            "shots": [256, 512],
            "score": [0.2, 0.9],
            TARGET: [0.6, 0.8],
        }
    )
    selected = select_configurations(ConfigScoreModel(), data, ScoreFeatureSet())
    by_budget = selected.set_index("budget")
    assert by_budget.loc["B256", "selected_config_id"] == "C01"
    assert by_budget.loc["B512", "selected_config_id"] == "C02"


def test_fixed_baseline_is_learned_from_training_rows_only() -> None:
    data = pd.DataFrame(
        {
            "noise_condition": ["N0", "N0"],
            "config_id": ["C01", "C02"],
            "shots": [256, 512],
            TARGET: [0.8, 0.9],
        }
    )
    choices = fixed_baseline_choices(data).set_index("budget")
    assert choices.loc["B256", "fixed_config_id"] == "C01"
    assert choices.loc["B512", "fixed_config_id"] == "C02"


def test_exact_sign_flip_is_two_sided_and_bounded() -> None:
    assert exact_sign_flip_pvalue([1.0, 1.0]) == pytest.approx(0.5)
    assert exact_sign_flip_pvalue([0.0, 0.0]) == pytest.approx(1.0)


def test_enhanced_features_reconstruct_manifest_graph() -> None:
    metadata = pd.DataFrame(
        [
            {
                "graph_id": "G0001",
                "graph_family": "erdos_renyi",
                "num_nodes": 10,
                "num_edges": 12,
                "graph_seed": 10000,
            }
        ]
    )
    features = build_enhanced_graph_features(metadata)
    assert features.loc[0, "average_degree"] == pytest.approx(2.4)
    assert features.loc[0, "component_count"] >= 1
    assert features.loc[0, "spectral_radius"] > 0
