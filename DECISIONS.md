# NQComp 2027 — Methodology Decisions

This file records the methodological decisions used for the frozen experiment and final ML analysis.

| ID | Date | Decision | Reason |
|---|---|---|---|
| D01 | 2026-09-17 | MaxCut is the core optimization problem. | Keeps the project focused on one reproducible combinatorial optimization task. |
| D02 | 2026-09-17 | Use a deterministic population of 100 graph instances. | Provides fixed graph identities for reproducible experiments and graph-level ML splitting. |
| D03 | 2026-09-17 | Use 12 frozen QAOA configurations: p={1,2,3} × {COBYLA,SPSA} × {256,512 shots}. | Provides a controlled configuration search space for the selector. |
| D04 | 2026-09-17 | Use N0 ideal simulation, N1 moderate controlled noise, and N2 higher controlled noise. | Enables controlled noise-aware evaluation. |
| D05 | 2026-09-17 | Use exact MaxCut for the supported small graphs. | Provides the reference optimum needed for normalized QAOA quality metrics. |
| D06 | 2026-09-18 | Random-regular graph degree is 4. | Keeps the regular-graph construction valid for all configured node counts, including 15 nodes. |
| D07 | 2026-09-18 | Resource budgets are derived from declared/measured execution constraints rather than creating duplicate QAOA runs. | Keeps budget conditioning from inflating the number of quantum experiments. |
| D08 | 2026-09-18 | Current raw experiments use two run seeds per graph/configuration/condition. | This is the seed protocol represented by the current frozen CSV results. |
| D09 | 2026-09-18 | N1/N2 evaluate parameters obtained from the corresponding N0 configuration evaluations instead of independently optimizing under noise. | Enables a controlled comparison of configuration behavior under different noise conditions. |
| D10 | 2026-09-18 | Mean expected approximation ratio is the primary quality metric; best sampled ratio is secondary. | Expected performance distinguishes configurations more reliably than a best-observed sample that can saturate the exact optimum. |
| D11 | 2026-09-18 | Raw quantum results are immutable after a dataset snapshot is frozen. | Preserves reproducibility and traceability. |
| D12 | 2026-09-18 | ML train/validation/test splitting is performed at the graph level. | Prevents leakage from repeated observations of the same graph. |
| D13 | 2026-09-18 | Features derived from target-graph QAOA outcomes are prohibited as ML inputs for that target graph. | Prevents target leakage in the selector evaluation. |
| D14 | 2026-09-18 | The original noisy snapshot covered only G0001-G0089. | Records the historical 89-graph analysis snapshot. |
| D15 | 2026-09-18 | G0090-G0100 were not represented as failed or zero-valued noisy observations in the historical snapshot. | They were simply outside that earlier N1/N2 dataset snapshot. |
| D16 | 2026-09-18 | GPU/CPU is an execution-environment choice rather than a change to the scientific experiment definition. | Preserves comparability while allowing available compute resources to be used. |
| D17 | 2026-09-18 | Completion of G0090-G0100 under N1/N2 is treated as a dataset extension/new snapshot rather than a silent edit of the historical 89-graph snapshot. | Preserves provenance. |
| D18 | 2026-09-18 | ML-derived datasets must be stored separately from raw result CSVs. | Keeps experimental evidence separate from transformations used for analysis. |
| D19 | 2026-09-29 | The completed N0/N1/N2 snapshot now contains all 100 graphs and 7200 raw observations. | Establishes the current primary dataset after completing the previously missing noisy evaluations. |
| D20 | 2026-09-29 | The current primary ML population is all 100 complete graphs. | Every graph now has identical N0/N1/N2 and 12-configuration coverage. |
| D21 | 2026-09-29 | Use a deterministic 70/15/15 graph-level split with seed 2027: 70 train, 15 validation, 15 test. | Provides a reproducible leakage-safe partition over the complete 100-graph population. |
| D22 | 2026-09-29 | The primary ML table contains 3600 seed-aggregated rows. | 100 graphs × 12 configurations × 3 noise conditions. |
| D23 | 2026-09-29 | Score candidate configurations by predicted mean expected approximation ratio. | Candidate descriptors are available before selection, while target-graph outcomes remain prohibited. |
| D24 | 2026-09-29 | Define B256/B512 as maximum shots-per-circuit feasibility constraints. | Provides explicit resource-aware candidate filtering without adding duplicate quantum runs. |
| D25 | 2026-09-29 | Select the model by mean validation selection regret; use row RMSE only as a secondary metric. | Optimizes the actual decision objective without using test data. |
| D26 | 2026-09-29 | Use ExtraTrees with the full pre-execution structural feature set for the current frozen final model. | ExtraTrees achieved the lowest validation selection regret in the current model comparison. |
| D27 | 2026-09-29 | Compare against a training-only fixed configuration for each noise/budget case, seeded random selection, and a feasible per-case oracle. | Separates deployable baselines from an unattainable upper reference. |
| D28 | 2026-09-29 | Retain model comparison, feature ablation, and selector-aware permutation importance as validation-only diagnostics. | Provides robustness and interpretability without test-set tuning. |
| D29 | 2026-09-29 | Treat the current 100-graph snapshot as the primary paper dataset; the historical 89-graph analysis remains provenance only. | Prevents mixing incompatible experimental snapshots in the final paper. |

## Current Primary Data Snapshot

```text
Graphs:              100
Configurations:       12
Noise conditions:      3
Run seeds:              2
Raw rows:           7200
Aggregated ML rows: 3600
```

Coverage:

```text
N0: 100 graphs / 2400 rows
N1: 100 graphs / 2400 rows
N2: 100 graphs / 2400 rows
```

## Current ML Split

```text
Common complete graphs: 100
Train:                  70
Validation:             15
Test:                   15
Graph overlap:           0
Split seed:           2027
```

Rows:

```text
Train:       2520
Validation:   540
Test:         540
Total:       3600
```

## Current Final ML Selector

The current selector uses ExtraTrees with the full pre-execution structural feature set. Model and feature selection are performed on validation data only using mean selection regret. The final model is refit on train + validation and evaluated once on the held-out test graphs.

The current test evaluation contains:

```text
15 graphs × 3 noise conditions × 2 budgets = 90 cases
```

Headline frozen test summary:

```text
Random expected mean:       0.694513
Fixed baseline mean:        0.707455
ML selector mean:           0.711856
Feasible oracle mean:       0.730775
ML − fixed:                 +0.004401
Relative gain:              +0.622148%
Mean regret:                 0.018918
Oracle selection accuracy:  22.2222%
Oracle top-3 accuracy:      61.1111%
```

These are descriptive empirical results on the current held-out test set and must not be interpreted as universal guarantees.

## Historical Snapshot

The repository also contains artifacts from the earlier 89-graph noisy snapshot and its 62/13/14 split. Those files are retained for provenance/reproducibility of the earlier analysis but are **not** the primary paper results.

Any future methodological change to the configuration registry, noise model, seed policy, utility, budget definition, leakage rules, split strategy, or primary metric must be recorded as a new decision before affected analyses are run.
