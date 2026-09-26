"""Validate the frozen QAOA raw datasets before ML preprocessing.

This script checks the data contract without modifying the raw CSV files.
Run from the repository root:
    python scripts/validate_ml_data.py
"""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
N0_PATH = RESULTS / "n0_results.csv"
NOISE_PATH = RESULTS / "noisy_results.csv"

EXPECTED_COLUMNS = {
    "expected_cut",
    "best_sampled_cut",
    "optimizer",
    "depth",
    "noise_condition",
    "optimizer_evaluations",
    "circuit_executions",
    "total_executed_shots",
    "two_qubit_gates",
    "circuit_depth",
    "simulator_runtime_seconds",
    "optimal_parameters",
    "experiment_id",
    "graph_id",
    "graph_family",
    "num_nodes",
    "num_edges",
    "graph_seed",
    "config_id",
    "shots_per_circuit",
    "exact_optimum",
    "run_seed",
    "schema_version",
    "expected_approximation_ratio",
    "best_sampled_approximation_ratio",
    "run_status",
    "failure_reason",
}

EXPECTED_CONFIGS = {f"C{i:02d}" for i in range(1, 13)}
EXPECTED_N0_GRAPHS = {f"G{i:04d}" for i in range(1, 101)}
EXPECTED_NOISY_GRAPHS = {f"G{i:04d}" for i in range(1, 90)}
EXPECTED_NOISE = {"N1", "N2"}
EXPECTED_SEEDS = {1000, 1001}
KEY = ["graph_id", "config_id", "noise_condition", "run_seed"]


def fail(errors, message):
    errors.append(message)
    print(f"❌ {message}")


def validate_common(df, name, errors):
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    missing = EXPECTED_COLUMNS - set(df.columns)
    extra = set(df.columns) - EXPECTED_COLUMNS
    if missing:
        fail(errors, f"{name}: missing columns: {sorted(missing)}")
    if extra:
        print(f"⚠️ {name}: extra columns: {sorted(extra)}")

    if "run_status" in df:
        status = df["run_status"].value_counts(dropna=False).to_dict()
        print(f"Run status: {status}")
        bad = df["run_status"].ne("success")
        if bad.any():
            fail(errors, f"{name}: {int(bad.sum())} rows are not successful")

    numeric_columns = [
        "expected_cut", "best_sampled_cut", "depth",
        "optimizer_evaluations", "circuit_executions",
        "total_executed_shots", "two_qubit_gates", "circuit_depth",
        "simulator_runtime_seconds", "num_nodes", "num_edges",
        "shots_per_circuit", "exact_optimum", "run_seed",
        "expected_approximation_ratio", "best_sampled_approximation_ratio",
    ]
    present_numeric = [c for c in numeric_columns if c in df.columns]
    if present_numeric:
        nonfinite = df[present_numeric].apply(pd.to_numeric, errors="coerce").isna().sum()
        bad_numeric = {k: int(v) for k, v in nonfinite.items() if v}
        if bad_numeric:
            fail(errors, f"{name}: non-numeric/NaN values: {bad_numeric}")

    if set(KEY).issubset(df.columns):
        dup = df.duplicated(KEY, keep=False)
        print(f"Duplicate logical keys: {int(dup.sum())}")
        if dup.any():
            fail(errors, f"{name}: duplicate logical keys found")

    if "config_id" in df:
        configs = set(df["config_id"].dropna().astype(str))
        print(f"Configurations: {sorted(configs)}")
        if configs != EXPECTED_CONFIGS:
            fail(errors, f"{name}: configuration set is not exactly C01-C12")

    if "run_seed" in df:
        seeds = set(pd.to_numeric(df["run_seed"], errors="coerce").dropna().astype(int))
        print(f"Run seeds: {sorted(seeds)}")
        if seeds != EXPECTED_SEEDS:
            fail(errors, f"{name}: seed set is not exactly {{1000, 1001}}")

    if "expected_approximation_ratio" in df and "expected_cut" in df and "exact_optimum" in df:
        denom = pd.to_numeric(df["exact_optimum"], errors="coerce")
        expected = pd.to_numeric(df["expected_cut"], errors="coerce") / denom
        recorded = pd.to_numeric(df["expected_approximation_ratio"], errors="coerce")
        mismatch = (expected - recorded).abs() > 1e-10
        mismatch &= denom.ne(0)
        if mismatch.any():
            fail(errors, f"{name}: approximation-ratio consistency failures: {int(mismatch.sum())}")


def validate_n0(df, errors):
    validate_common(df, "N0", errors)
    if "noise_condition" in df:
        values = set(df["noise_condition"].dropna().astype(str))
        print(f"Noise conditions: {sorted(values)}")
        if values != {"N0"}:
            fail(errors, f"N0: unexpected noise conditions {sorted(values)}")
    if "graph_id" in df:
        graphs = set(df["graph_id"].dropna().astype(str))
        print(f"Graphs: {len(graphs)} ({min(graphs)} → {max(graphs)})")
        if graphs != EXPECTED_N0_GRAPHS:
            fail(errors, "N0: graph coverage is not exactly G0001-G0100")
    expected_rows = 100 * 12 * 2
    if len(df) != expected_rows:
        fail(errors, f"N0: expected {expected_rows} rows, found {len(df)}")
    if set(KEY).issubset(df.columns):
        counts = df.groupby("graph_id").size()
        bad = counts[counts != 24]
        if not bad.empty:
            fail(errors, f"N0: graphs without exactly 24 rows: {bad.to_dict()}")


def validate_noise(df, errors):
    validate_common(df, "N1/N2", errors)
    if "noise_condition" in df:
        values = set(df["noise_condition"].dropna().astype(str))
        print(f"Noise conditions: {sorted(values)}")
        if values != EXPECTED_NOISE:
            fail(errors, f"N1/N2: unexpected noise conditions {sorted(values)}")
    if "graph_id" in df:
        graphs = set(df["graph_id"].dropna().astype(str))
        print(f"Graphs: {len(graphs)} ({min(graphs)} → {max(graphs)})")
        if graphs != EXPECTED_NOISY_GRAPHS:
            fail(errors, "N1/N2: graph coverage is not exactly G0001-G0089")
    expected_rows = 89 * 12 * 2 * 2
    if len(df) != expected_rows:
        fail(errors, f"N1/N2: expected {expected_rows} rows, found {len(df)}")
    if set(KEY).issubset(df.columns):
        counts = df.groupby(["graph_id", "noise_condition"]).size()
        bad = counts[counts != 24]
        if not bad.empty:
            fail(errors, f"N1/N2: graph-condition groups without exactly 24 rows: {bad.to_dict()}")


def main():
    errors = []

    if not N0_PATH.exists():
        fail(errors, f"Missing file: {N0_PATH}")
    if not NOISE_PATH.exists():
        fail(errors, f"Missing file: {NOISE_PATH}")
    if errors:
        print("\nValidation cannot start because required files are missing.")
        return 1

    n0 = pd.read_csv(N0_PATH)
    noise = pd.read_csv(NOISE_PATH)

    validate_n0(n0, errors)
    validate_noise(noise, errors)

    print(f"\n{'=' * 72}\nCROSS-DATASET CHECKS\n{'=' * 72}")
    n0_graphs = set(n0["graph_id"].astype(str))
    noise_graphs = set(noise["graph_id"].astype(str))
    print(f"N0 graphs: {len(n0_graphs)}")
    print(f"Noisy graphs: {len(noise_graphs)}")
    print(f"Common graphs: {len(n0_graphs & noise_graphs)}")
    print(f"Noisy-only graphs: {sorted(noise_graphs - n0_graphs)}")
    print(f"N0-only graphs: {sorted(n0_graphs - noise_graphs)}")

    if not noise_graphs.issubset(n0_graphs):
        fail(errors, "Noisy dataset contains graph IDs absent from N0")

    if set(noise["graph_id"].astype(str)) != EXPECTED_NOISY_GRAPHS:
        fail(errors, "Noisy graph coverage differs from the frozen G0001-G0089 snapshot")

    if errors:
        print(f"\n❌ VALIDATION FAILED — {len(errors)} issue(s)")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        return 1

    print("\n✅ ALL ML DATA VALIDATION CHECKS PASSED")
    print("Raw CSV files were read only; no data was modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
