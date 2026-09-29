# NQComp 2027 — Paper Results

This directory contains paper-ready tables and figures derived from:

`data/ml/final_selector/`

The **current primary paper snapshot** is the completed 100-graph experiment.
The earlier 89-graph bundle is retained only for historical provenance and must
not be mixed into the primary manuscript.

## Current protocol

- Split: 70 train / 15 validation / 15 test graphs
- Test graphs: 15
- Test cases: 90 graph/noise/budget cases
- Noise conditions: N0, N1, N2
- Budgets: B256 and B512
- Model selected on validation only: ExtraTrees
- Final fit: train + validation
- Test partition: used only for final evaluation
- Split unit: graph
- Split seed: 2027
- Raw observations: 7200
- Seed-aggregated ML rows: 3600

## Main current result

| Method | Mean expected approximation ratio |
|---|---:|
| Uniform-random expectation | 0.694513 |
| Training-only fixed baseline | 0.707455 |
| ML selector (ExtraTrees) | 0.711856 |
| Feasible oracle | 0.730775 |

ML minus fixed baseline:

**+0.004401 absolute / +0.622148% relative**

Mean feasible-oracle regret:

**0.018918**

Oracle selection accuracy:

**22.22%**

Top-3 oracle coverage:

**61.11%**

## Budget results

### B256

- ML: 0.706733
- Fixed: 0.705794
- ML − Fixed: +0.000939
- Relative change: +0.133%

### B512

- ML: 0.716979
- Fixed: 0.709116
- ML − Fixed: +0.007863
- Relative change: +1.109%

## Noise results

### N0

- ML: 0.730574
- Fixed: 0.726060
- ML − Fixed: +0.004513
- Relative change: +0.622%

### N1

- ML: 0.711370
- Fixed: 0.706140
- ML − Fixed: +0.005230
- Relative change: +0.741%

### N2

- ML: 0.693626
- Fixed: 0.690164
- ML − Fixed: +0.003461
- Relative change: +0.501%

## Budget × noise

The case-level table `T4_budget_noise_results` should be used for the full
interaction analysis. The strongest aggregate gain occurs at B512, while the
B256 result is smaller. The paper should report these differences directly
rather than collapsing all resource conditions into one universal claim.

## Statistical interpretation

The observed overall ML improvement is descriptive. The current graph-clustered
statistics report a 95% bootstrap interval for the overall ML-minus-fixed
difference of approximately **[-0.00308, 0.01231]** with an exact sign-flip
p-value of **0.3071**. The manuscript should therefore avoid describing the
overall gain as statistically established.

Some budget/noise slices show different behavior; those values should be
reported from the frozen tables without post-hoc selection of favorable cases.

## File guide

The current generation script produces:

- T1 main results
- T2 budget results
- T3 noise results
- T4 budget × noise results
- T5 validation model comparison
- T6 feature ablation
- T7 permutation importance
- T8 statistical tests
- T9 selected configurations
- T10 fixed baselines
- T11 graph features
- T12 experimental metadata

Figures F1–F10 provide method comparison, budget/noise behavior, model
validation, feature ablation, permutation importance, regret distribution,
ML-vs-oracle behavior, configuration selection frequency, and ML gain over the
fixed baseline.

## Source of truth

The canonical frozen analysis remains:

`data/ml/final_selector/`

This directory is a paper-facing copy/consolidation layer. Raw QAOA
experimental CSVs are not modified by this process.

No model retraining or test-set model selection is performed while generating
the paper bundle.
