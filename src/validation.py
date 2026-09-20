"""Validation utilities for the NQComp quantum raw dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .configurations import CONFIGURATIONS
from .noise import NOISE_CONDITIONS


SCHEMA_VERSION = "1.0"

REQUIRED_COLUMNS = {
    "experiment_id",
    "graph_id",
    "graph_family",
    "num_nodes",
    "num_edges",
    "graph_seed",
    "noise_condition",
    "config_id",
    "depth",
    "optimizer",
    "shots_per_circuit",
    "expected_cut",
    "best_sampled_cut",
    "exact_optimum",
    "expected_approximation_ratio",
    "best_sampled_approximation_ratio",
    "optimizer_evaluations",
    "circuit_executions",
    "total_executed_shots",
    "two_qubit_gates",
    "circuit_depth",
    "simulator_runtime_seconds",
    "run_seed",
    "run_status",
    "failure_reason",
    "schema_version",
}

EXPECTED_CONFIG_IDS = {
    config.config_id for config in CONFIGURATIONS
}

EXPECTED_NOISE_IDS = set(NOISE_CONDITIONS)

EXPECTED_SEEDS = {20000, 20001, 20002}

EXPECTED_GRAPH_FAMILIES = {
    "erdos_renyi",
    "random_regular",
}

MAX_CIRCUIT_EXECUTIONS = 200


class DatasetValidationError(ValueError):
    """Raised when a raw experiment dataset violates the data contract."""


def validate_raw_results(results: pd.DataFrame) -> None:
    """Validate a raw quantum experiment DataFrame.

    Raises DatasetValidationError when the dataset violates the
    frozen quantum-to-ML data contract.
    """
    _validate_columns(results)
    _validate_allowed_values(results)
    _validate_experiment_ids(results)
    _validate_resources(results)
    _validate_measurements(results)
    _validate_ratios(results)
    _validate_failures(results)
    _validate_schema_version(results)


def validate_raw_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate a raw CSV.

    Returns the loaded DataFrame if validation succeeds.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    results = pd.read_csv(path, dtype={"schema_version": str})
    validate_raw_results(results)

    return results


def _validate_columns(results: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(results.columns)

    if missing:
        raise DatasetValidationError(
            f"Missing required columns: {sorted(missing)}"
        )


def _validate_allowed_values(results: pd.DataFrame) -> None:
    if not set(results["config_id"].dropna()).issubset(
        EXPECTED_CONFIG_IDS
    ):
        raise DatasetValidationError("Unknown config_id detected")

    if not set(results["noise_condition"].dropna()).issubset(
        EXPECTED_NOISE_IDS
    ):
        raise DatasetValidationError("Unknown noise_condition detected")

    if not set(results["graph_family"].dropna()).issubset(
        EXPECTED_GRAPH_FAMILIES
    ):
        raise DatasetValidationError("Unknown graph_family detected")

    if not set(results["run_seed"].dropna()).issubset(
        EXPECTED_SEEDS
    ):
        raise DatasetValidationError("Unexpected run_seed detected")


def _validate_experiment_ids(results: pd.DataFrame) -> None:
    if results["experiment_id"].duplicated().any():
        raise DatasetValidationError(
            "Duplicate experiment_id detected"
        )

    keys = [
        "graph_id",
        "config_id",
        "noise_condition",
        "run_seed",
    ]

    if results.duplicated(subset=keys).any():
        raise DatasetValidationError(
            "Duplicate raw-record grain detected"
        )


def _validate_resources(results: pd.DataFrame) -> None:
    successful = results["run_status"].eq("success")

    executions = results.loc[successful, "circuit_executions"]

    if (executions > MAX_CIRCUIT_EXECUTIONS).any():
        raise DatasetValidationError(
            "Circuit execution limit exceeded"
        )

    shots = results.loc[successful, "shots_per_circuit"]
    allowed_shots = {256, 512}

    if not set(shots).issubset(allowed_shots):
        raise DatasetValidationError(
            "Unexpected shots_per_circuit value"
        )

    expected_total = (
        results.loc[successful, "shots_per_circuit"]
        * results.loc[successful, "circuit_executions"]
    )

    actual_total = results.loc[
        successful, "total_executed_shots"
    ]

    if not (expected_total == actual_total).all():
        raise DatasetValidationError(
            "total_executed_shots accounting mismatch"
        )


def _validate_measurements(results: pd.DataFrame) -> None:
    successful = results["run_status"].eq("success")

    required = [
        "expected_cut",
        "best_sampled_cut",
        "exact_optimum",
        "expected_approximation_ratio",
        "best_sampled_approximation_ratio",
        "circuit_executions",
        "total_executed_shots",
    ]

    if results.loc[successful, required].isna().any().any():
        raise DatasetValidationError(
            "Successful row contains missing measurements"
        )


def _validate_ratios(results: pd.DataFrame) -> None:
    successful = results["run_status"].eq("success")

    expected_ratio = (
        results.loc[successful, "expected_cut"]
        / results.loc[successful, "exact_optimum"]
    )

    actual_ratio = results.loc[
        successful, "expected_approximation_ratio"
    ]

    if not (
        (expected_ratio - actual_ratio).abs() < 1e-9
    ).all():
        raise DatasetValidationError(
            "expected_approximation_ratio mismatch"
        )

    best_ratio = (
        results.loc[successful, "best_sampled_cut"]
        / results.loc[successful, "exact_optimum"]
    )

    actual_best_ratio = results.loc[
        successful, "best_sampled_approximation_ratio"
    ]

    if not (
        (best_ratio - actual_best_ratio).abs() < 1e-9
    ).all():
        raise DatasetValidationError(
            "best_sampled_approximation_ratio mismatch"
        )


def _validate_failures(results: pd.DataFrame) -> None:
    allowed_statuses = {"success", "failed"}

    if not set(results["run_status"].dropna()).issubset(
        allowed_statuses
    ):
        raise DatasetValidationError(
            "Unknown run_status detected"
        )

    failed = results["run_status"].eq("failed")

    if results.loc[failed, "failure_reason"].fillna("").eq("").any():
        raise DatasetValidationError(
            "Failed row is missing failure_reason"
        )


def _validate_schema_version(results: pd.DataFrame) -> None:
    versions = {
        str(version)
        for version in results["schema_version"].dropna()
    }

    if versions != {SCHEMA_VERSION}:
        raise DatasetValidationError(
            f"Expected schema version {SCHEMA_VERSION}, "
            f"found {sorted(versions)}"
        )