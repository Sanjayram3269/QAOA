DATA_SCHEMA.md
# NQComp 2027 — Data Schema

## 1. Purpose

This document defines the structure and interpretation of the raw QAOA experimental result data.

Canonical raw files:

```text
data/results/n0_results.csv
data/results/noisy_results.csv
```

Raw files contain per-evaluation observations and are treated as immutable experimental artifacts after a snapshot is frozen.

## 2. Raw Observation

The raw experiment records one observation for:

```text
graph × QAOA configuration × run seed × noise condition
```

For N0 the condition is ideal/noiseless.

For N1/N2 the corresponding noise condition is explicitly recorded.

## 3. Configuration Dimensions

The frozen configuration registry contains 12 configurations:

```text
depth ∈ {1, 2, 3}
optimizer ∈ {COBYLA, SPSA}
shots ∈ {256, 512}
```

The configuration IDs are C01-C12.

## 4. Seeds

The current frozen dataset uses two run seeds per graph/configuration/condition.

Therefore:

```text
12 configurations × 2 seeds = 24 rows
```

per graph and condition.

No three-seed claim should be made for the current raw dataset.

## 5. Dataset Coverage

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
89 graphs
G0001 → G0089
12 configurations
2 seeds
2136 rows
```

### N2

```text
89 graphs
G0001 → G0089
12 configurations
2 seeds
2136 rows
```

Combined raw data:

```text
6672 rows
```

## 6. Current Missing Coverage

The noisy dataset does not currently contain N1/N2 observations for:

```text
G0090 → G0100
```

These graphs are not treated as failures or zero-performance observations.

They are simply outside the current noisy dataset snapshot.

## 7. Core Raw Fields

The current schema includes fields representing:

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

## 8. Approximation Ratio

The primary QAOA quality quantity is expected approximation ratio:

```text
expected_approximation_ratio
=
expected_cut / exact_optimum
```

Best sampled approximation ratio is retained as a secondary descriptive metric.

## 9. Seed Aggregation

When ML preprocessing aggregates repeated runs, aggregation should occur after separating:

```text
graph_id
config_id
noise_condition
```

The raw two-seed observations must remain available and unchanged.

The aggregation rule must be documented by the ML preprocessing implementation.

## 10. Resource Budgets

If resource-budget labels are derived, they must be constructed from the measured raw execution data rather than creating duplicate QAOA runs.

Budget definitions must be consistent with the frozen experiment decisions.

## 11. ML Leakage Policy

The ML selector can use only information available before selecting a configuration for the target graph.

Allowed categories include:
- graph features
- known noise condition
- declared resource constraint

Forbidden leakage sources include:
- target graph QAOA performance
- exact optimum of the target graph
- configuration outcome metrics
- utility calculated from target-graph outcomes
- runtime measured after executing a target configuration
- post-execution measurements

## 12. Graph-Level Splitting

The split unit is the graph.

A graph must occur in only one of:

```text
TRAIN
VALIDATION
TEST
```

All available observations for the same graph must remain in that same split.

This rule applies across N0, N1, N2, configurations, and seeds.

## 13. Derived ML Data

Derived data must be stored separately from the raw results.

Recommended location:

```text
data/ml/
```

Examples:

```text
data/ml/features.csv
data/ml/targets.csv
data/ml/predictions.csv
```

Do not overwrite the raw experimental CSV files.

## 14. Integrity Checks

Before ML processing, verify:

```text
N0 = 2400 rows, 100 graphs
N1 = 2136 rows, 89 graphs
N2 = 2136 rows, 89 graphs
```

Also verify:
- N1 and N2 cover the same 89 graph IDs
- each noisy graph has 48 rows total
- each condition has 24 rows per graph
- no unexpected duplicate `(graph_id, config_id, noise_condition, run_seed)` keys exist
- run_status values are valid
- required fields are present

## 15. Missing Data Policy

A missing graph/condition is not automatically a failed experiment.

For the current snapshot, G0090-G0100 are outside N1/N2 coverage.

ML analyses requiring noisy labels must either:
- operate on the 89 available noisy graphs, or
- use a later dataset version that includes the additional noisy evaluations.

The selected policy must be stated in the analysis.

## 16. Versioning

Raw data corrections or extensions should create a new explicit dataset version.

Do not silently edit old experimental rows.

## 17. Reproducibility

The raw data should remain traceable to:
- graph manifest
- graph-generation seeds
- configuration registry
- run seeds
- noise model
- experiment runner
- code revision
