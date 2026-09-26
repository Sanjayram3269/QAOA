DECISIONS.md
# NQComp 2027 — Methodology Decisions

This file records the methodological decisions used for the current frozen experiment.

| ID | Date | Decision | Reason |
|---|---|---|---|
| D01 | 2026-09-17 | MaxCut is the core optimization problem. | Keeps the project focused on one reproducible combinatorial optimization task. |
| D02 | 2026-09-17 | Use a deterministic population of 100 graph instances. | Provides fixed graph identities for reproducible experiments and graph-level ML splitting. |
| D03 | 2026-09-17 | Use 12 frozen QAOA configurations: p={1,2,3} × {COBYLA,SPSA} × {256,512 shots}. | Provides a controlled configuration search space for the selector. |
| D04 | 2026-09-17 | Use N0 ideal simulation, N1 moderate controlled noise, and N2 higher controlled noise. | Enables controlled noise-aware evaluation. |
| D05 | 2026-09-17 | Use exact MaxCut for the supported small graphs. | Provides the reference optimum needed for normalized QAOA quality metrics. |
| D06 | 2026-09-18 | Random-regular graph degree is 4. | Keeps the regular-graph construction valid for all configured node counts, including 15 nodes. |
| D07 | 2026-09-18 | Resource budgets are derived from measured execution data rather than by creating duplicate QAOA runs. | Keeps budget conditioning from inflating the number of quantum experiments. |
| D08 | 2026-09-18 | Current raw experiments use two run seeds per graph/configuration/condition. | This is the seed protocol represented by the frozen CSV results. |
| D09 | 2026-09-18 | N1/N2 evaluate parameters obtained from the corresponding N0 configuration evaluations instead of independently optimizing under noise. | Enables a controlled comparison of configuration behavior under different noise conditions. |
| D10 | 2026-09-18 | Mean expected approximation ratio is the primary quality metric; best sampled ratio is secondary. | Expected performance distinguishes configurations more reliably than a best-observed sample that can saturate the exact optimum. |
| D11 | 2026-09-18 | Raw quantum results are immutable after a dataset snapshot is frozen. | Preserves reproducibility and traceability. |
| D12 | 2026-09-18 | ML train/validation/test splitting is performed at the graph level. | Prevents leakage from repeated observations of the same graph. |
| D13 | 2026-09-18 | Features derived from target-graph QAOA outcomes are prohibited as ML inputs for that target graph. | Prevents target leakage in the selector evaluation. |
| D14 | 2026-09-18 | Current N0 coverage is G0001-G0100; current N1/N2 coverage is G0001-G0089. | Records the actual frozen experimental coverage. |
| D15 | 2026-09-18 | G0090-G0100 are not represented as failed or zero-valued noisy observations. | They are simply not part of the current N1/N2 dataset snapshot. |
| D16 | 2026-09-18 | GPU/CPU is an execution-environment choice rather than a change to the scientific experiment definition. | Preserves comparability while allowing available compute resources to be used. |
| D17 | 2026-09-18 | Future completion of G0090-G0100 under N1/N2 must be documented as an extension/new dataset update. | Avoids silently changing the frozen dataset. |
| D18 | 2026-09-18 | ML-derived datasets must be stored separately from raw result CSVs. | Keeps experimental evidence separate from transformations used for analysis. |

## Current Frozen Data Snapshot

```text
N0:
100 graphs
2400 rows

N1:
89 graphs
2136 rows

N2:
89 graphs
2136 rows

TOTAL:
6672 rows
```

## Current ML Starting Point

The ML stage begins from the frozen raw quantum results and proceeds through:

```text
validation
→ aggregation
→ feature engineering
→ target/utility construction
→ graph-level splitting
→ baselines
→ ML selector
→ unseen-graph evaluation
```

Any later methodological change to the configuration registry, noise model, seed policy, utility, budget definition, leakage rules, split strategy, or primary metric must be recorded as a new decision before affected analyses are run.
