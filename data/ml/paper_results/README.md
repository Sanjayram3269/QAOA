# NQComp 2027 — Paper Results

This directory contains the frozen, paper-ready tables derived from:

`data/ml/final_selector/`

These files are intended for preparing the paper's Results, Method Evaluation,
Ablation, Statistical Analysis, and Figure sections.

## Frozen protocol

- Split: 62 train / 13 validation / 14 test graphs
- Test graphs: 14
- Test cases: 84 graph/noise/budget cases
- Noise conditions: N0, N1, N2
- Budgets: B256 and B512
- Model selected on validation only: ExtraTrees
- Final fit: train + validation
- Test partition: used only for final evaluation
- Split unit: graph
- Seed: 2027

## Main frozen result

| Method | Mean expected approximation ratio |
|---|---:|
| Uniform-random expectation | 0.689134 |
| Training-only fixed baseline | 0.700960 |
| ML selector | 0.702228 |
| Feasible oracle | 0.724351 |

ML minus fixed baseline:

**+0.001269 absolute / +0.181% relative**

Mean feasible-oracle regret:

**0.022123**

Oracle selection accuracy:

**17.86%**

Top-3 oracle coverage:

**58.33%**

## Budget results

### B256

- ML: 0.697375
- Fixed: 0.699103
- ML − Fixed: −0.001728
- Relative change: −0.247%

### B512

- ML: 0.707081
- Fixed: 0.702816
- ML − Fixed: +0.004265
- Relative change: +0.607%

## Noise results

### N0

- ML: 0.722956
- Fixed: 0.718739
- ML − Fixed: +0.004217
- Relative change: +0.587%

### N1

- ML: 0.698641
- Fixed: 0.700436
- ML − Fixed: −0.001796
- Relative change: −0.256%

### N2

- ML: 0.685088
- Fixed: 0.683703
- ML − Fixed: +0.001385
- Relative change: +0.203%

## Statistical interpretation

The observed overall improvement of the ML selector over the fixed baseline is
descriptive. The frozen analysis reports graph-clustered uncertainty and
hypothesis tests; these should be used directly when writing the paper rather
than presenting the 0.181% improvement as a statistically established effect.

## File guide

| File | Purpose |
|---|---|
| `01_main_results.csv` | Main paper comparison |
| `02_budget_results.csv` | B256/B512 comparison |
| `03_noise_results.csv` | N0/N1/N2 comparison |
| `04_budget_noise_results.csv` | Budget × noise analysis |
| `05_model_validation_comparison.csv` | Validation-only model selection |
| `06_feature_ablation.csv` | Graph-feature ablation |
| `07_permutation_importance.csv` | Validation interpretability |
| `08_statistical_tests.csv` | Graph-clustered statistical analysis |
| `09_test_selected_configurations.csv` | Case-level frozen test decisions |
| `10_fixed_baseline_choices.csv` | Training-only fixed baseline |
| `11_test_graph_features.csv` | Frozen graph features |

## Source of truth

The canonical frozen analysis remains:

`data/ml/final_selector/`

This directory is a paper-facing copy/consolidation layer only.

Raw QAOA experimental CSVs are not modified by this process.

No model was retrained and no test-set model selection was performed while
creating these paper tables.
