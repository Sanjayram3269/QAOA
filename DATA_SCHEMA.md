# Data Schema

This document is the frozen interface between Sanjay's quantum pipeline and Neha's ML pipeline.

## Raw record grain

One raw row represents one unique:

```text
(graph_id, config_id, noise_condition, run_seed)
```

Resource-budget cases are derived during label construction. They do not create additional QAOA runs:

- **B256:** only configurations with `shots_per_circuit <= 256` are feasible.
- **B512:** configurations with `shots_per_circuit <= 512` are feasible.

## Required raw fields

| Field | Type | Meaning |
|---|---|---|
| experiment_id | string | `G0001_C01_N0_S20000` style unique identifier |
| graph_id | string | Stable graph identifier |
| graph_family | string | `erdos_renyi` or `random_regular` |
| num_nodes | int | Number of vertices |
| num_edges | int | Number of edges |
| graph_seed | int | Graph-generation seed |
| noise_condition | string | N0, N1, or N2 |
| config_id | string | Frozen ID C01-C12 |
| depth | int | QAOA depth |
| optimizer | string | COBYLA or SPSA |
| shots_per_circuit | int | 256 or 512 |
| expected_cut | float/null | Expected cut from the final measured distribution |
| best_sampled_cut | int/null | Highest observed sampled cut |
| exact_optimum | int | Exact Max-Cut optimum |
| expected_approximation_ratio | float/null | `expected_cut / exact_optimum` |
| best_sampled_approximation_ratio | float/null | `best_sampled_cut / exact_optimum` |
| optimizer_evaluations | int/null | Objective circuits used during optimization |
| circuit_executions | int/null | Optimizer evaluations plus final evaluation |
| total_executed_shots | int/null | `shots_per_circuit * circuit_executions` |
| two_qubit_gates | int/null | All two-qubit instructions in the final compiled circuit |
| circuit_depth | int/null | Final compiled circuit depth |
| simulator_runtime_seconds | float/null | Simulator wall-clock runtime |
| run_seed | int | Paired repetition seed |
| run_status | string | `success` or `failed` |
| failure_reason | string | Empty on success; diagnostic text on failure |
| schema_version | string | Contract version, initially `1.0` |

`optimal_parameters` may be retained as an optional diagnostic field. It is never an ML input.

## Seed aggregation

Feasibility experiments use three paired seeds: 20000, 20001, and 20002. All configurations for the same graph and noise condition use the same seed set.

Aggregate successful rows by:

```text
(graph_id, config_id, noise_condition)
```

The primary quality is the mean `expected_approximation_ratio`. Best-sampled quality is secondary because it frequently reaches the exact optimum even when expected performance differs.

The primary cost is mean `total_executed_shots`.

## Utility and labels

The frozen primary utility is:

```text
Q_norm = mean expected_approximation_ratio
C_norm = min(mean total_executed_shots / 102400, 1)
U = 0.8 * Q_norm - 0.2 * C_norm
```

Sensitivity analysis uses alpha values 0.6, 0.7, 0.8, and 0.9.

For each `(graph_id, noise_condition, resource_budget)`, choose the highest-utility feasible candidate. Candidates within 0.005 utility of the maximum are practically tied. Break practical ties by:

1. Lower mean total executed shots
2. Lower depth
3. Fewer shots per circuit
4. Lexicographically smaller configuration ID

## Leakage rule

The ML selector may use only information available before configuration selection:

- Graph features
- Known noise condition
- Declared resource budget

Forbidden ML inputs include configuration fields, QAOA outcomes, exact optimum, utility, oracle label, runtime, circuit counts, and all post-execution measurements.

## Split rule

Split unique graphs 70/15/15 using split seed 2027, balanced by graph family and size where feasible, before expanding noise, budget, configuration, or seed rows. No graph may occur in more than one split.

## Immutability and failures

Raw files are immutable after generation. Corrections create a new dataset/schema version. Failed rows remain in the raw data with `run_status=failed`; they are not silently deleted or converted into successful observations.
