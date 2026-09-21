"""Training, constrained prediction, baselines, and evaluation for the selector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .configurations import CONFIG_BY_ID
from .features import MODEL_FEATURE_COLUMNS
from .ml_dataset import ALPHA, BUDGETS, COST_NORMALIZER


CATEGORICAL_FEATURES = ["graph_family", "noise_condition", "resource_budget"]
NUMERIC_FEATURES = [
    name for name in MODEL_FEATURE_COLUMNS if name not in CATEGORICAL_FEATURES
]
KEY_COLUMNS = ["graph_id", "noise_condition", "resource_budget"]


@dataclass
class SelectorExperimentResult:
    selected_model_name: str
    selected_model: Pipeline
    validation_metrics: pd.DataFrame
    test_metrics: dict[str, float | int]
    baseline_metrics: dict[str, float | int]
    test_predictions: pd.DataFrame
    baseline_predictions: pd.DataFrame


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_models(seed: int = 2027) -> dict[str, Pipeline]:
    """Return the two frozen candidate ML selectors."""
    return {
        "random_forest": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def predict_feasible(
    model: Pipeline,
    features: pd.DataFrame,
    top_k: int = 3,
) -> pd.DataFrame:
    """Rank model classes after masking configurations outside each budget."""
    probabilities = model.predict_proba(features[MODEL_FEATURE_COLUMNS])
    classes = np.asarray(model.classes_, dtype=str)
    rows: list[dict] = []

    for position, (_, row) in enumerate(features.reset_index(drop=True).iterrows()):
        budget = str(row["resource_budget"])
        if budget not in BUDGETS:
            raise ValueError(f"Unknown resource budget: {budget}")
        feasible = np.array(
            [CONFIG_BY_ID[config_id].shots <= BUDGETS[budget] for config_id in classes]
        )
        if not feasible.any():
            raise ValueError(f"Model has no feasible learned class for {budget}")

        scores = np.where(feasible, probabilities[position], -np.inf)
        order = np.argsort(-scores)
        ranked = [classes[index] for index in order if np.isfinite(scores[index])]
        rows.append(
            {
                **{column: row[column] for column in KEY_COLUMNS},
                "predicted_config_id": ranked[0],
                "top_k_config_ids": "|".join(ranked[:top_k]),
            }
        )
    return pd.DataFrame(rows)


def build_candidate_table(aggregated: pd.DataFrame) -> pd.DataFrame:
    """Expand aggregate configuration outcomes across feasible budgets."""
    rows: list[pd.DataFrame] = []
    for budget_name, shot_limit in BUDGETS.items():
        feasible = aggregated[aggregated["shots_per_circuit"] <= shot_limit].copy()
        feasible["resource_budget"] = budget_name
        feasible["candidate_quality"] = feasible[
            "mean_expected_approximation_ratio"
        ]
        feasible["candidate_cost"] = feasible["mean_total_executed_shots"]
        feasible["candidate_utility"] = (
            ALPHA * feasible["candidate_quality"]
            - (1.0 - ALPHA)
            * (feasible["candidate_cost"] / COST_NORMALIZER).clip(upper=1.0)
        )
        rows.append(feasible)
    return pd.concat(rows, ignore_index=True)


def evaluate_predictions(
    predictions: pd.DataFrame,
    labelled_features: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Evaluate class selection and the utility achieved by each prediction."""
    truth_columns = KEY_COLUMNS + [
        "oracle_config_id",
        "oracle_utility",
        "oracle_quality",
        "oracle_cost",
    ]
    scored = predictions.merge(
        labelled_features[truth_columns],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    candidate_lookup = candidates[
        KEY_COLUMNS
        + ["config_id", "candidate_utility", "candidate_quality", "candidate_cost"]
    ].rename(columns={"config_id": "predicted_config_id"})
    scored = scored.merge(
        candidate_lookup,
        on=KEY_COLUMNS + ["predicted_config_id"],
        how="left",
        validate="one_to_one",
    )
    if scored["candidate_utility"].isna().any():
        raise ValueError("A predicted configuration is missing or budget-infeasible")

    scored["correct"] = scored["predicted_config_id"] == scored["oracle_config_id"]
    scored["utility_regret"] = (
        scored["oracle_utility"] - scored["candidate_utility"]
    ).clip(lower=0.0)
    scored["top_3_correct"] = [
        oracle in ranking.split("|")
        for oracle, ranking in zip(
            scored["oracle_config_id"],
            scored["top_k_config_ids"],
        )
    ]

    metrics: dict[str, float | int] = {
        "examples": int(len(scored)),
        "exact_accuracy": float(scored["correct"].mean()),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                scored["oracle_config_id"], scored["predicted_config_id"]
            )
        ),
        "macro_f1": float(
            f1_score(
                scored["oracle_config_id"],
                scored["predicted_config_id"],
                average="macro",
                zero_division=0,
            )
        ),
        "top_3_accuracy": float(scored["top_3_correct"].mean()),
        "mean_utility_regret": float(scored["utility_regret"].mean()),
        "median_utility_regret": float(scored["utility_regret"].median()),
        "mean_selected_quality": float(scored["candidate_quality"].mean()),
        "mean_selected_cost": float(scored["candidate_cost"].mean()),
    }
    return metrics, scored


def condition_global_best_predictions(
    candidates: pd.DataFrame,
    fit_graph_ids: set[str],
    evaluation_features: pd.DataFrame,
) -> pd.DataFrame:
    """Fit the frozen condition-aware global-best baseline without test data."""
    fit = candidates[candidates["graph_id"].isin(fit_graph_ids)].copy()
    if fit.empty:
        raise ValueError("No candidate rows are available to fit the baseline")

    summary = (
        fit.groupby(
            ["noise_condition", "resource_budget", "config_id", "depth", "shots_per_circuit"],
            as_index=False,
        )
        .agg(
            mean_utility=("candidate_utility", "mean"),
            mean_cost=("candidate_cost", "mean"),
        )
    )

    winners: list[dict] = []
    for (noise, budget), group in summary.groupby(
        ["noise_condition", "resource_budget"], sort=True
    ):
        maximum = group["mean_utility"].max()
        tied = group[group["mean_utility"] >= maximum - 0.005].sort_values(
            ["mean_cost", "depth", "shots_per_circuit", "config_id"]
        )
        winners.append(
            {
                "noise_condition": noise,
                "resource_budget": budget,
                "predicted_config_id": tied.iloc[0]["config_id"],
            }
        )

    predictions = evaluation_features[KEY_COLUMNS].merge(
        pd.DataFrame(winners),
        on=["noise_condition", "resource_budget"],
        how="left",
        validate="many_to_one",
    )
    if predictions["predicted_config_id"].isna().any():
        raise ValueError("Baseline has no winner for an evaluation condition")
    predictions["top_k_config_ids"] = predictions["predicted_config_id"]
    return predictions


def run_selector_experiment(
    dataset_with_splits: pd.DataFrame,
    aggregated: pd.DataFrame,
    seed: int = 2027,
) -> SelectorExperimentResult:
    """Select a model on validation, refit on development data, and test once."""
    partitions = {
        name: dataset_with_splits[dataset_with_splits["split"] == name].copy()
        for name in ("train", "validation", "test")
    }
    if any(frame.empty for frame in partitions.values()):
        raise ValueError("Train, validation, and test partitions must all be non-empty")

    candidates = build_candidate_table(aggregated)
    validation_rows: list[dict] = []
    fitted: dict[str, Pipeline] = {}
    for name, model in build_models(seed).items():
        model.fit(
            partitions["train"][MODEL_FEATURE_COLUMNS],
            partitions["train"]["oracle_config_id"],
        )
        fitted[name] = model
        predictions = predict_feasible(model, partitions["validation"])
        metrics, _ = evaluate_predictions(
            predictions, partitions["validation"], candidates
        )
        validation_rows.append({"model": name, **metrics})

    validation_metrics = pd.DataFrame(validation_rows).sort_values(
        ["mean_utility_regret", "exact_accuracy", "model"],
        ascending=[True, False, True],
    )
    selected_name = str(validation_metrics.iloc[0]["model"])

    development = pd.concat(
        [partitions["train"], partitions["validation"]], ignore_index=True
    )
    selected_model = build_models(seed)[selected_name]
    selected_model.fit(
        development[MODEL_FEATURE_COLUMNS], development["oracle_config_id"]
    )

    test_predictions = predict_feasible(selected_model, partitions["test"])
    test_metrics, test_scored = evaluate_predictions(
        test_predictions, partitions["test"], candidates
    )
    baseline_predictions = condition_global_best_predictions(
        candidates=candidates,
        fit_graph_ids=set(development["graph_id"]),
        evaluation_features=partitions["test"],
    )
    baseline_metrics, baseline_scored = evaluate_predictions(
        baseline_predictions, partitions["test"], candidates
    )

    return SelectorExperimentResult(
        selected_model_name=selected_name,
        selected_model=selected_model,
        validation_metrics=validation_metrics.reset_index(drop=True),
        test_metrics=test_metrics,
        baseline_metrics=baseline_metrics,
        test_predictions=test_scored,
        baseline_predictions=baseline_scored,
    )
