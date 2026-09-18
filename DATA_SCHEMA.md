# Data Schema — Quantum → ML Contract

This document is the interface between Sanjay's quantum pipeline and Neha's ML pipeline.

## One row

One row represents one `(graph_id, config_id, noise_condition, run_seed)` evaluation.

## Required fields

| Field | Type | Meaning |
|---|---|---|
| graph_id | string | Stable graph identifier, e.g. G0001 |
| graph_family | string | `erdos_renyi` or `random_regular` |
| num_nodes | int | Number of vertices |
| num_edges | int | Number of edges |
| graph_seed | int | Seed used to generate the graph |
| noise_condition | string | N0, N1, or N2 |
| config_id | string | C01–C12 |
| depth | int | QAOA depth p |
| optimizer | string | COBYLA or SPSA |
| shots | int | 256 or 512 |
| expected_cut | float | Expected cut value from measured distribution |
| best_sampled_cut | int | Highest cut value observed in the sampled bitstrings |
| exact_optimum | int | Exact Max-Cut optimum for the graph |
| expected_approximation_ratio | float | expected_cut / exact_optimum |
| best_sampled_approximation_ratio | float | best_sampled_cut / exact_optimum |
| optimizer_evaluations | int | Objective evaluations used |
| two_qubit_gates | int | CX count after transpilation |
| circuit_depth | int | Transpiled circuit depth |
| simulator_runtime_seconds | float | Simulator wall-clock runtime |
| run_seed | int | Reproducibility seed |

## Leakage rule

The ML selector may use only information available **before** selecting a QAOA configuration. Graph structure/features and known execution constraints are allowed. QAOA outcomes, exact optimum, utility, regret, runtime after execution, and selected configuration fields must not be used as predictive features for configuration selection.

## Split rule

Train/validation/test splits must be made at the **graph level**, never by randomly splitting rows after configuration/noise expansion. The same graph must not appear in more than one split.

## Primary handoff

Sanjay exports raw quantum results. Neha derives features, utility/oracle labels, ML datasets, predictions, baselines, and analysis without modifying the raw records.
