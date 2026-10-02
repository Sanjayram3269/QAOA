# Final ML Selector Method

## Scope and data boundary

The current primary selector consumes only
`data/ml/splits/ml_performance_common.csv`, the frozen derived table covering
the **100 graphs** complete under N0, N1, and N2. The canonical raw QAOA result
CSVs are not read or modified by the finalization script.

The current deterministic graph-level split uses seed 2027:

| Partition | Graphs | Rows |
|---|---:|---:|
| Train | 70 | 2520 |
| Validation | 15 | 540 |
| Test | 15 | 540 |

All rows for one graph remain in one partition across configurations and noise
conditions. There is zero graph overlap between partitions.

## Prediction formulation

One supervised row represents a candidate
`(graph, noise condition, QAOA configuration)`. The regression target is mean
expected approximation ratio, aggregated over the two run seeds. The model
scores every feasible candidate and recommends the candidate with the largest
predicted ratio within the active resource budget.

Only pre-selection information is used:

- graph family, size, edge count, and density;
- deterministic structural descriptors reconstructed from the frozen graph
  seed (degree statistics, triangles, clustering, transitivity, connectivity,
  and spectral descriptors);
- known noise condition; and
- declared candidate descriptors (depth, optimizer, shots, and configuration
  ID).

The exact optimum, QAOA outcomes, target-graph runtime, and post-execution
measurements are not selector features.

## Resource budgets

Budgets are feasibility constraints on shots per circuit:

- B256 permits configurations with at most 256 shots per circuit.
- B512 permits configurations with at most 512 shots per circuit.

This definition avoids treating N0 and N1/N2 as identical total-work
experiments when noisy conditions reuse the corresponding N0 optimized
parameters.

## Model and feature selection

Ridge, random forest, ExtraTrees, and histogram gradient boosting are fit on
the 70 training graphs. The primary validation criterion is mean selection
regret over validation `(graph, noise, budget)` cases:

```text
regret = feasible oracle quality - selected quality
```

Row-level MAE/RMSE/R² are secondary diagnostics. Feature ablations are also
evaluated only on validation data. The current frozen model choice is
**ExtraTrees** with the full pre-execution structural feature set. It is then
refit on train plus validation. The 15 held-out test graphs are touched only
after model and feature selection are frozen.

## Baselines and metrics

The principal fixed baseline chooses one configuration for each
`(noise condition, budget)` using training graphs only. A seeded random
selection reference and the feasible per-case oracle provide lower/upper
references.

Reported metrics include mean selected quality, absolute and relative gain
over fixed, mean regret, exact oracle selection accuracy, top-3 oracle
coverage, and row-level MAE/RMSE/R².

The current frozen test evaluation contains:

```text
15 test graphs × 3 noise conditions × 2 budgets = 90 cases
```

## Current headline test result

The current final-selector run reports:

| Method | Mean expected approximation ratio |
|---|---:|
| Uniform-random expectation | 0.694513 |
| Training-only fixed baseline | 0.707455 |
| ML selector (ExtraTrees) | 0.711856 |
| Feasible oracle | 0.730775 |

The corresponding ML-minus-fixed gain is `+0.004401` (`+0.622148%`), with
mean feasible-oracle regret `0.018918`, exact oracle-selection accuracy
`22.2222%`, and top-3 oracle coverage `61.1111%`.

These are empirical frozen-test results and should be interpreted as such;
they are not universal-performance guarantees.

## Interpretability

Selector-aware grouped permutation importance is computed on validation data,
not test data. Features are permuted coherently at their natural unit (graph,
noise label, or configuration), and importance is the resulting increase in
selection regret. This answers how feature groups affect the selection
decision rather than only row-wise prediction error.

## Reproduction

```bash
python scripts/validate_ml_data.py
python scripts/build_ml_dataset.py
python scripts/create_ml_splits.py
python scripts/finalize_ml_selector.py
```

Paper outputs can then be regenerated with:

```bash
python scripts/generate_paper_outputs.py
```

Outputs are written to `data/ml/final_selector/` and
`data/ml/paper_results/`. The raw QAOA CSVs are never modified by these
analysis stages.
