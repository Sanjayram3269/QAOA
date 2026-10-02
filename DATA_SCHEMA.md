# NQComp 2027 — Data Schema

## 1. Purpose

This document defines the structure and interpretation of the QAOA experimental result data and the derived ML dataset.

Canonical raw files:

```text
data/results/n0_results.csv
data/results/noisy_results.csv
```

Raw files are treated as immutable experimental artifacts after a snapshot is frozen. Derived ML data are stored separately under `data/ml/`.

## 2. Raw Observation

The raw experiment records one observation for:

```text
graph × QAOA configuration × run seed × noise condition
```

N0 is ideal/noiseless. N1 and N2 are explicitly recorded controlled-noise conditions.

## 3. Configuration Dimensions

The frozen configuration registry contains 12 configurations:

```text
depth ∈ {1, 2, 3}
optimizer ∈ {COBYLA, SPSA}
shots ∈ {256, 512}
```

Configuration IDs are C01–C12.

## 4. Seeds

The current frozen dataset uses two run seeds per graph/configuration/condition.

Therefore:

```text
12 configurations × 2 seeds = 24 raw rows
```

per graph and condition.

## 5. Current Dataset Coverage

The current completed snapshot has complete coverage for all 100 graphs under all three conditions.

### N0

```text
100 graphs
G0001 → G0100
12 configurations
2 seeds
2400 rows
```

### N1

```text
100 graphs
G0001 → G0100
12 configurations
2 seeds
2400 rows
```

### N2

```text
100 graphs
G0001 → G0100
12 configurations
2 seeds
2400 rows
```

Combined raw data:

```text
7200 rows
```

After seed aggregation:

```text
100 graphs × 12 configurations × 3 conditions = 3600 ML rows
```

## 6. Core Raw Fields

The schema includes fields representing:

### Identification
- experiment_id
- graph_id
- graph_family
- graph_seed
- schema_version

### Graph structure
- num_nodes
- num_edges

### QAOA configuration
- config_id
- depth
- optimizer
- shots_per_circuit

### Experimental condition
- noise_condition
- run_seed

### Quality
- expected_cut
- best_sampled_cut
- exact_optimum
- expected_approximation_ratio
- best_sampled_approximation_ratio

### Resource/execution
- optimizer_evaluations
- circuit_executions
- total_executed_shots
- two_qubit_gates
- circuit_depth
- simulator_runtime_seconds

### Run state
- run_status
- failure_reason

`optimal_parameters` may be retained as a diagnostic field but is not an ML input.

## 7. Approximation Ratio

The primary QAOA quality quantity is expected approximation ratio:

```text
expected_approximation_ratio
=
expected_cut / exact_optimum
```

Best sampled approximation ratio is retained as a secondary descriptive metric.

## 8. Seed Aggregation

ML preprocessing aggregates repeated runs after separating:

```text
graph_id
config_id
noise_condition
```

The two-seed raw observations remain available and unchanged. The primary ML target is the mean expected approximation ratio.

## 9. Resource Budgets

The selector uses shot-feasibility budgets derived from the declared configuration shots:

```text
B256: candidate shots per circuit <= 256
B512: candidate shots per circuit <= 512
```

Budget labels are feasibility constraints rather than duplicated quantum experiments.

## 10. ML Leakage Policy

The ML selector can use only information available before selecting a configuration for the target graph.

Allowed categories include:
- graph features;
- known noise condition;
- declared configuration/resource descriptors.

Forbidden leakage sources include:
- target-graph QAOA performance;
- exact target-graph optimum;
- configuration outcome metrics;
- utility calculated from target-graph outcomes;
- runtime measured after executing a target configuration;
- post-execution measurements.

## 11. Graph-Level Splitting

The split unit is the graph.

A graph must occur in only one of:

```text
TRAIN
VALIDATION
TEST
```

All observations for the same graph remain in that partition across configurations, noise conditions, and seeds.

Current split:

```text
100 common graphs
70 train
15 validation
15 test
seed = 2027
```

This produces:

```text
2520 train rows
540 validation rows
540 test rows
3600 rows total
```

with zero graph overlap.

## 12. Derived ML Data

Derived data are stored separately from raw results:

```text
data/ml/
```

Important current files include:

```text
data/ml/ml_performance.csv
data/ml/splits/ml_performance_common.csv
data/ml/final_selector/
data/ml/final_analysis/
data/ml/paper_results/
```

Raw experimental CSVs must never be overwritten by ML transformations.

## 13. Integrity Checks

For the current complete snapshot, verify:

```text
N0 = 2400 rows, 100 graphs
N1 = 2400 rows, 100 graphs
N2 = 2400 rows, 100 graphs
TOTAL = 7200 raw rows
ML = 3600 aggregated rows
```

Also verify:
- all three conditions cover G0001-G0100;
- each graph/condition has 12 configurations;
- each configuration/condition has two run seeds;
- no unexpected duplicate `(graph_id, config_id, noise_condition, run_seed)` keys exist;
- required fields are present;
- graph-level split overlap is zero.

## 14. Missing Data Policy

The current primary snapshot has no missing graph/condition coverage. Any future extension or correction must be versioned explicitly rather than silently changing this snapshot.

## 15. Versioning

Raw-data corrections or extensions should create a new explicit dataset version.

Do not silently edit historical experimental rows.

## 16. Reproducibility

The raw data should remain traceable to:
- graph manifest;
- graph-generation seeds;
- configuration registry;
- run seeds;
- noise model;
- experiment runner;
- code revision;
- derived-dataset manifest;
- graph split manifest;
- final selector analysis manifest.
