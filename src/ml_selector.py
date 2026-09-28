"""Leakage-safe, resource-aware ML selection for the frozen QAOA dataset."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 2027
TARGET = "mean_expected_approximation_ratio"
BUDGETS = {"B256": 256, "B512": 512}
KEY_COLUMNS = ["graph_id", "noise_condition", "budget"]

BASE_GRAPH_FEATURES = ["num_nodes", "num_edges", "density"]
ENHANCED_GRAPH_FEATURES = [
    "average_degree",
    "degree_std",
    "min_degree",
    "max_degree",
    "triangle_count",
    "average_clustering",
    "transitivity",
    "component_count",
    "largest_component_fraction",
    "spectral_radius",
    "adjacency_spectral_gap",
    "algebraic_connectivity",
    "laplacian_spectral_radius",
    "triangle_density",
    "degree_assortativity",
    "average_shortest_path_length",
    "diameter",
]
GRAPH_FEATURES = BASE_GRAPH_FEATURES + ENHANCED_GRAPH_FEATURES
CONFIG_NUMERIC = ["depth", "shots"]
CONFIG_CATEGORICAL = ["optimizer", "config_id"]
CONTEXT_CATEGORICAL = ["graph_family", "noise_condition"]
FULL_NUMERIC = CONFIG_NUMERIC + GRAPH_FEATURES
FULL_CATEGORICAL = CONFIG_CATEGORICAL + CONTEXT_CATEGORICAL
FULL_FEATURES = FULL_NUMERIC + FULL_CATEGORICAL


@dataclass(frozen=True)
class FeatureSet:
    name: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def columns(self) -> list[str]:
        return list(self.numeric + self.categorical)


FEATURE_SETS = {
    "full": FeatureSet(
        "full",
        tuple(FULL_NUMERIC),
        tuple(FULL_CATEGORICAL),
    ),
    "without_graph_structure": FeatureSet(
        "without_graph_structure",
        tuple(CONFIG_NUMERIC),
        tuple(CONFIG_CATEGORICAL + ["noise_condition"]),
    ),
    "without_noise_condition": FeatureSet(
        "without_noise_condition",
        tuple(FULL_NUMERIC),
        tuple(CONFIG_CATEGORICAL + ["graph_family"]),
    ),
    "without_config_id": FeatureSet(
        "without_config_id",
        tuple(CONFIG_NUMERIC + GRAPH_FEATURES),
        tuple(["optimizer", "graph_family", "noise_condition"]),
    ),
}


def validate_split_integrity(data: pd.DataFrame) -> None:
    """Validate the frozen graph split and complete candidate coverage."""
    required = {
        "graph_id",
        "config_id",
        "noise_condition",
        "split",
        "shots",
        TARGET,
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing ML columns: {sorted(missing)}")

    graph_splits = data[["graph_id", "split"]].drop_duplicates()
    if graph_splits["graph_id"].duplicated().any():
        raise ValueError("Graph leakage: a graph occurs in multiple splits")

    expected_splits = {"train", "validation", "test"}
    if set(graph_splits["split"]) != expected_splits:
        raise ValueError("Expected train, validation, and test graph splits")

    complete = data.groupby(["graph_id", "noise_condition"])["config_id"].nunique()
    if not (complete == 12).all():
        raise ValueError("Incomplete 12-configuration coverage detected")

    if data.duplicated(["graph_id", "config_id", "noise_condition"]).any():
        raise ValueError("Duplicate aggregate candidate rows detected")


def build_enhanced_graph_features(graph_metadata: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct deterministic graphs and compute pre-execution features."""
    import networkx as nx

    from .graph_generation import GraphSpec, generate_graph

    required = {
        "graph_id",
        "graph_family",
        "num_nodes",
        "num_edges",
        "graph_seed",
    }
    missing = required - set(graph_metadata.columns)
    if missing:
        raise ValueError(f"Missing graph metadata: {sorted(missing)}")

    metadata_columns = [
        "graph_id",
        "graph_family",
        "num_nodes",
        "num_edges",
        "graph_seed",
    ]
    unique = graph_metadata[metadata_columns].drop_duplicates()
    if unique["graph_id"].duplicated().any():
        raise ValueError("Inconsistent graph metadata for a graph_id")

    rows: list[dict] = []
    for row in unique.sort_values("graph_id").itertuples(index=False):
        spec = GraphSpec(
            graph_id=str(row.graph_id),
            family=str(row.graph_family),
            num_nodes=int(row.num_nodes),
            seed=int(row.graph_seed),
            probability=0.35 if row.graph_family == "erdos_renyi" else None,
            degree=4 if row.graph_family == "random_regular" else None,
        )
        graph = generate_graph(spec)
        if graph.number_of_edges() != int(row.num_edges):
            raise ValueError(
                f"Reconstructed edge count mismatch for {row.graph_id}: "
                f"{graph.number_of_edges()} != {int(row.num_edges)}"
            )

        degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
        components = [len(component) for component in nx.connected_components(graph)]
        adjacency = nx.to_numpy_array(graph, dtype=float)
        adjacency_eigenvalues = np.linalg.eigvalsh(adjacency)
        spectral_radius = float(adjacency_eigenvalues[-1])
        adjacency_spectral_gap = float(
            adjacency_eigenvalues[-1] - adjacency_eigenvalues[-2]
        )
        laplacian = np.diag(degrees) - adjacency
        laplacian_eigenvalues = np.linalg.eigvalsh(laplacian)
        algebraic_connectivity = float(max(0.0, laplacian_eigenvalues[1]))
        laplacian_spectral_radius = float(laplacian_eigenvalues[-1])
        triangle_count = int(sum(nx.triangles(graph).values()) // 3)

        edge_u, edge_v = np.where(np.triu(adjacency, k=1) > 0)
        endpoint_u = degrees[edge_u]
        endpoint_v = degrees[edge_v]
        assortativity_x = np.concatenate([endpoint_u, endpoint_v])
        assortativity_y = np.concatenate([endpoint_v, endpoint_u])
        if np.std(assortativity_x) == 0 or np.std(assortativity_y) == 0:
            degree_assortativity = 0.0
        else:
            degree_assortativity = float(
                np.corrcoef(assortativity_x, assortativity_y)[0, 1]
            )

        largest_component = max(
            nx.connected_components(graph), key=len
        )
        pair_distances: list[int] = []
        for source in sorted(largest_component):
            distances = {source: 0}
            queue = [source]
            for node in queue:
                for neighbor in np.flatnonzero(adjacency[node]):
                    neighbor = int(neighbor)
                    if neighbor not in largest_component or neighbor in distances:
                        continue
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
            pair_distances.extend(
                distance
                for target, distance in distances.items()
                if target > source
            )
        average_shortest_path_length = float(np.mean(pair_distances))
        diameter = int(max(pair_distances))
        rows.append(
            {
                "graph_id": str(row.graph_id),
                "average_degree": float(degrees.mean()),
                "degree_std": float(degrees.std(ddof=0)),
                "min_degree": int(degrees.min()),
                "max_degree": int(degrees.max()),
                "triangle_count": triangle_count,
                "average_clustering": float(nx.average_clustering(graph)),
                "transitivity": float(nx.transitivity(graph)),
                "component_count": int(len(components)),
                "largest_component_fraction": float(
                    max(components) / graph.number_of_nodes()
                ),
                "spectral_radius": spectral_radius,
                "adjacency_spectral_gap": adjacency_spectral_gap,
                "algebraic_connectivity": algebraic_connectivity,
                "laplacian_spectral_radius": laplacian_spectral_radius,
                "triangle_density": float(
                    triangle_count
                    / max(1.0, graph.number_of_nodes() * (graph.number_of_nodes() - 1) * (graph.number_of_nodes() - 2) / 6)
                ),
                "degree_assortativity": degree_assortativity,
                "average_shortest_path_length": average_shortest_path_length,
                "diameter": diameter,
            }
        )
    return pd.DataFrame(rows)


def merge_graph_features(
    performance: pd.DataFrame,
    enhanced_features: pd.DataFrame,
) -> pd.DataFrame:
    """Join deterministic graph features to candidate performance rows."""
    if enhanced_features["graph_id"].duplicated().any():
        raise ValueError("Enhanced features must have one row per graph")
    merged = performance.merge(
        enhanced_features,
        on="graph_id",
        how="left",
        validate="many_to_one",
    )
    if merged[ENHANCED_GRAPH_FEATURES].isna().any().any():
        raise ValueError("Missing enhanced graph features after join")
    return merged


def _preprocessor(feature_set: FeatureSet, dense: bool = False) -> ColumnTransformer:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    return ColumnTransformer(
        [("categorical", encoder, list(feature_set.categorical))],
        remainder="passthrough",
        sparse_threshold=0.0 if dense else 0.3,
    )


def make_model(
    model_name: str,
    feature_set: FeatureSet,
    seed: int = SEED,
) -> Pipeline:
    """Construct one deterministic candidate-performance regressor."""
    if model_name == "ridge":
        return Pipeline(
            [
                ("preprocess", _preprocessor(feature_set)),
                ("scale", StandardScaler(with_mean=False)),
                ("model", Ridge(alpha=1.0)),
            ]
        )
    if model_name == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=1,
        )
        return Pipeline(
            [("preprocess", _preprocessor(feature_set)), ("model", estimator)]
        )
    if model_name == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=1,
        )
        return Pipeline(
            [("preprocess", _preprocessor(feature_set)), ("model", estimator)]
        )
    if model_name == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        )
        return Pipeline(
            [("preprocess", _preprocessor(feature_set, dense=True)), ("model", estimator)]
        )
    raise ValueError(f"Unknown model: {model_name}")


def row_metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict:
    actual = np.asarray(list(actual), dtype=float)
    predicted = np.asarray(list(predicted), dtype=float)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def select_configurations(
    model: Pipeline,
    data: pd.DataFrame,
    feature_set: FeatureSet,
) -> pd.DataFrame:
    """Select the highest-predicted feasible candidate for each budget."""
    candidates = data[
        ["graph_id", "noise_condition", "config_id", "shots", TARGET]
    ].copy()
    candidates["feasibility_shots"] = (
        data["_feasibility_shots"].to_numpy()
        if "_feasibility_shots" in data
        else data["shots"].to_numpy()
    )
    candidates["predicted_quality"] = model.predict(data[feature_set.columns])

    selections: list[pd.DataFrame] = []
    for budget, limit in BUDGETS.items():
        feasible = candidates[candidates["feasibility_shots"] <= limit].copy()
        group = ["graph_id", "noise_condition"]
        feasible = feasible.sort_values(group + ["predicted_quality", "config_id"], ascending=[True, True, False, True])
        feasible["predicted_rank"] = feasible.groupby(group).cumcount() + 1
        selected = feasible.loc[
            feasible.groupby(group)["predicted_quality"].idxmax()
        ].copy()
        selected = selected.rename(
            columns={
                "config_id": "selected_config_id",
                TARGET: "selected_quality",
            }
        )
        oracle = feasible.loc[feasible.groupby(group)[TARGET].idxmax()].copy()
        oracle = oracle[group + ["config_id", TARGET]].rename(
            columns={"config_id": "oracle_config_id", TARGET: "oracle_quality"}
        )
        selected = selected.merge(oracle, on=group, validate="one_to_one")
        oracle_ranks = feasible[group + ["config_id", "predicted_rank"]].rename(
            columns={"config_id": "oracle_config_id", "predicted_rank": "oracle_predicted_rank"}
        )
        selected = selected.merge(
            oracle_ranks,
            on=group + ["oracle_config_id"],
            validate="one_to_one",
        )
        selected["budget"] = budget
        selected["regret"] = (
            selected["oracle_quality"] - selected["selected_quality"]
        ).clip(lower=0.0)
        selected["selected_is_oracle"] = (
            selected["selected_config_id"] == selected["oracle_config_id"]
        )
        selected["oracle_in_top3"] = selected["oracle_predicted_rank"] <= 3
        selections.append(selected)
    return pd.concat(selections, ignore_index=True)


def selection_metrics(selections: pd.DataFrame) -> dict:
    return {
        "selection_mean": float(selections["selected_quality"].mean()),
        "oracle_mean": float(selections["oracle_quality"].mean()),
        "mean_regret": float(selections["regret"].mean()),
        "median_regret": float(selections["regret"].median()),
        "oracle_selection_accuracy": float(selections["selected_is_oracle"].mean()),
        "oracle_top3_accuracy": float(selections["oracle_in_top3"].mean()),
        "cases": int(len(selections)),
    }


def compare_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_set: FeatureSet = FEATURE_SETS["full"],
    seed: int = SEED,
) -> tuple[pd.DataFrame, str, dict[str, Pipeline]]:
    """Choose a model using validation selection regret, never test data."""
    fitted: dict[str, Pipeline] = {}
    rows: list[dict] = []
    for name in (
        "ridge",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
    ):
        model = make_model(name, feature_set, seed)
        model.fit(train[feature_set.columns], train[TARGET])
        row_prediction = model.predict(validation[feature_set.columns])
        metrics = row_metrics(validation[TARGET], row_prediction)
        metrics.update(selection_metrics(select_configurations(model, validation, feature_set)))
        rows.append({"model": name, **metrics})
        fitted[name] = model

    comparison = pd.DataFrame(rows).sort_values(
        ["mean_regret", "oracle_selection_accuracy", "rmse", "model"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    return comparison, str(comparison.iloc[0]["model"]), fitted


def validation_ablations(
    model_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    rows: list[dict] = []
    for name, feature_set in FEATURE_SETS.items():
        model = make_model(model_name, feature_set, seed)
        model.fit(train[feature_set.columns], train[TARGET])
        metrics = row_metrics(
            validation[TARGET], model.predict(validation[feature_set.columns])
        )
        metrics.update(selection_metrics(select_configurations(model, validation, feature_set)))
        rows.append({"ablation": name, **metrics})
    return pd.DataFrame(rows).sort_values("mean_regret").reset_index(drop=True)


def fixed_baseline_choices(train: pd.DataFrame) -> pd.DataFrame:
    """Choose one configuration per noise/budget using train graphs only."""
    rows: list[dict] = []
    for budget, limit in BUDGETS.items():
        for noise in sorted(train["noise_condition"].unique()):
            feasible = train[
                (train["noise_condition"] == noise) & (train["shots"] <= limit)
            ]
            scores = (
                feasible.groupby("config_id", as_index=False)[TARGET]
                .mean()
                .sort_values([TARGET, "config_id"], ascending=[False, True])
            )
            rows.append(
                {
                    "budget": budget,
                    "noise_condition": noise,
                    "fixed_config_id": str(scores.iloc[0]["config_id"]),
                    "train_mean_quality": float(scores.iloc[0][TARGET]),
                }
            )
    return pd.DataFrame(rows)


def score_test_selections(
    ml_selections: pd.DataFrame,
    test: pd.DataFrame,
    fixed_choices: pd.DataFrame,
) -> pd.DataFrame:
    """Add paired fixed, random-expectation, and oracle references."""
    rows: list[pd.DataFrame] = []
    for budget, limit in BUDGETS.items():
        feasible = test[test["shots"] <= limit].copy()
        fixed = feasible.merge(
            fixed_choices[fixed_choices["budget"] == budget],
            on="noise_condition",
            validate="many_to_one",
        )
        fixed = fixed[fixed["config_id"] == fixed["fixed_config_id"]][
            ["graph_id", "noise_condition", "fixed_config_id", TARGET]
        ].rename(columns={TARGET: "fixed_quality"})
        random_expected = (
            feasible.groupby(["graph_id", "noise_condition"], as_index=False)[TARGET]
            .mean()
            .rename(columns={TARGET: "random_expected_quality"})
        )
        part = ml_selections[ml_selections["budget"] == budget].merge(
            fixed,
            on=["graph_id", "noise_condition"],
            validate="one_to_one",
        )
        part = part.merge(
            random_expected,
            on=["graph_id", "noise_condition"],
            validate="one_to_one",
        )
        part["ml_minus_fixed"] = part["selected_quality"] - part["fixed_quality"]
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def summarize_test(scored: pd.DataFrame) -> pd.DataFrame:
    scopes: list[tuple[str, pd.Series]] = [("ALL", pd.Series(True, index=scored.index))]
    scopes.extend((budget, scored["budget"].eq(budget)) for budget in BUDGETS)
    scopes.extend(
        (noise, scored["noise_condition"].eq(noise))
        for noise in sorted(scored["noise_condition"].unique())
    )
    scopes.extend(
        (
            f"{budget}_{noise}",
            scored["budget"].eq(budget) & scored["noise_condition"].eq(noise),
        )
        for budget in BUDGETS
        for noise in sorted(scored["noise_condition"].unique())
    )

    rows: list[dict] = []
    for scope, mask in scopes:
        part = scored[mask]
        rows.append(
            {
                "scope": scope,
                "graphs": int(part["graph_id"].nunique()),
                "cases": int(len(part)),
                "ml_mean": float(part["selected_quality"].mean()),
                "fixed_mean": float(part["fixed_quality"].mean()),
                "random_expected_mean": float(part["random_expected_quality"].mean()),
                "oracle_mean": float(part["oracle_quality"].mean()),
                "ml_minus_fixed": float(part["ml_minus_fixed"].mean()),
                "relative_gain_percent": float(
                    100.0 * part["ml_minus_fixed"].mean() / part["fixed_quality"].mean()
                ),
                "mean_regret": float(part["regret"].mean()),
                "oracle_selection_accuracy": float(part["selected_is_oracle"].mean()),
                "oracle_top3_accuracy": float(part["oracle_in_top3"].mean()),
            }
        )
    return pd.DataFrame(rows)


def exact_sign_flip_pvalue(values: Iterable[float]) -> float:
    """Two-sided exact paired randomization p-value for up to 20 graph units."""
    differences = np.asarray(list(values), dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0:
        return float("nan")
    observed = abs(float(differences.mean()))
    if len(differences) > 20:
        raise ValueError("Exact sign-flip enumeration is limited to 20 units")
    extreme = 0
    total = 0
    for signs in product((-1.0, 1.0), repeat=len(differences)):
        statistic = abs(float(np.mean(differences * np.asarray(signs))))
        extreme += int(statistic >= observed - 1e-15)
        total += 1
    return float(extreme / total)


def graph_clustered_paired_stats(
    scored: pd.DataFrame,
    seed: int = SEED,
    bootstrap_repetitions: int = 10000,
) -> pd.DataFrame:
    """Compute graph-clustered CIs and exact sign-flip tests for ML vs fixed."""
    scopes: list[tuple[str, pd.DataFrame]] = [("ALL", scored)]
    scopes.extend((budget, scored[scored["budget"] == budget]) for budget in BUDGETS)
    scopes.extend(
        (noise, scored[scored["noise_condition"] == noise])
        for noise in sorted(scored["noise_condition"].unique())
    )
    scopes.extend(
        (
            f"{budget}_{noise}",
            scored[
                (scored["budget"] == budget)
                & (scored["noise_condition"] == noise)
            ],
        )
        for budget in BUDGETS
        for noise in sorted(scored["noise_condition"].unique())
    )

    rows: list[dict] = []
    for offset, (scope, part) in enumerate(scopes):
        per_graph = part.groupby("graph_id")["ml_minus_fixed"].mean().to_numpy()
        rng = np.random.default_rng(seed + offset)
        indices = rng.integers(
            0, len(per_graph), size=(bootstrap_repetitions, len(per_graph))
        )
        bootstrap_means = per_graph[indices].mean(axis=1)
        rows.append(
            {
                "scope": scope,
                "graphs": int(len(per_graph)),
                "mean_ml_minus_fixed": float(per_graph.mean()),
                "median_ml_minus_fixed": float(np.median(per_graph)),
                "bootstrap_ci95_low": float(np.quantile(bootstrap_means, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(bootstrap_means, 0.975)),
                "exact_sign_flip_pvalue": exact_sign_flip_pvalue(per_graph),
                "ml_better_graph_fraction": float((per_graph > 0).mean()),
            }
        )
    result = pd.DataFrame(rows)
    condition_mask = result["scope"].str.match(r"B(256|512)_N[012]")
    pvalues = result.loc[condition_mask, "exact_sign_flip_pvalue"].to_numpy()
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(pvalues) - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    result["holm_adjusted_pvalue"] = np.nan
    result.loc[condition_mask, "holm_adjusted_pvalue"] = adjusted
    return result


def _permuted_copy(
    data: pd.DataFrame,
    columns: list[str],
    rng: np.random.Generator,
    unit: str,
) -> pd.DataFrame:
    result = data.copy()
    if "shots" in columns:
        result["_feasibility_shots"] = result["shots"]
    if unit == "graph":
        unique = result[["graph_id"] + columns].drop_duplicates("graph_id")
        permuted = unique[columns].iloc[rng.permutation(len(unique))].reset_index(drop=True)
        mapping = unique[["graph_id"]].reset_index(drop=True).join(permuted)
        result = result.drop(columns=columns).merge(mapping, on="graph_id", validate="many_to_one")
        return result
    if unit == "graph_condition":
        # Relabel conditions with one global bijection. This preserves the
        # candidate set for every graph/condition selection problem.
        labels = np.asarray(sorted(result["noise_condition"].unique()))
        mapping = dict(zip(labels, labels[rng.permutation(len(labels))]))
        result["noise_condition"] = result["noise_condition"].map(mapping)
        return result
    if unit == "configuration":
        descriptors = ["config_id", "depth", "shots", "optimizer"]
        unique = result[descriptors].drop_duplicates("config_id").sort_values("config_id")
        if columns == ["config_id"]:
            mapping = dict(
                zip(
                    unique["config_id"],
                    unique["config_id"].iloc[rng.permutation(len(unique))],
                )
            )
            result["config_id"] = result["config_id"].map(mapping)
            return result
        permuted = unique[columns].iloc[rng.permutation(len(unique))].reset_index(drop=True)
        mapping = unique[["config_id"]].reset_index(drop=True).join(permuted)
        result = result.drop(columns=columns).merge(
            mapping, on="config_id", validate="many_to_one"
        )
        return result
    raise ValueError(f"Unknown permutation unit: {unit}")


def selector_permutation_importance(
    model: Pipeline,
    validation: pd.DataFrame,
    feature_set: FeatureSet,
    seed: int = SEED,
    repetitions: int = 30,
) -> pd.DataFrame:
    """Measure validation regret increase using coherent feature permutations."""
    baseline = selection_metrics(
        select_configurations(model, validation, feature_set)
    )["mean_regret"]
    groups: list[tuple[str, list[str], str]] = [
        (name, [name], "graph") for name in ["graph_family"] + GRAPH_FEATURES
    ]
    groups.extend(
        [
            ("all_graph_structure", ["graph_family"] + GRAPH_FEATURES, "graph"),
            ("noise_condition", ["noise_condition"], "graph_condition"),
            (
                "configuration_descriptors",
                ["depth", "shots", "optimizer"],
                "configuration",
            ),
            ("config_id", ["config_id"], "configuration"),
        ]
    )

    rows: list[dict] = []
    for offset, (name, columns, unit) in enumerate(groups):
        values: list[float] = []
        for repeat in range(repetitions):
            rng = np.random.default_rng(seed + 1000 * offset + repeat)
            permuted = _permuted_copy(validation, columns, rng, unit)
            regret = selection_metrics(
                select_configurations(model, permuted, feature_set)
            )["mean_regret"]
            values.append(regret - baseline)
        rows.append(
            {
                "feature_group": name,
                "baseline_validation_regret": baseline,
                "mean_regret_increase": float(np.mean(values)),
                "std_regret_increase": float(np.std(values, ddof=1)),
                "repetitions": repetitions,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mean_regret_increase", ascending=False
    ).reset_index(drop=True)
