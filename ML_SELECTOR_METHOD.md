# Final ML Selector Method

## Scope and data boundary

The final selector consumes only
`data/ml/splits/ml_performance_common.csv`, the frozen derived table covering
the 89 graphs complete under N0, N1, and N2. The canonical raw QAOA result
CSVs are not read or modified by the finalization script.

The frozen graph-level split (seed 2027) is retained:

| Partition | Graphs |
|---|---:|
| Train | 62 |
| Validation | 13 |
| Test | 14 |

All rows for one graph remain in one partition across configurations and noise
conditions. Because preliminary test results already existed before this final
analysis, the split is not changed. This avoids selecting a more favorable
test set after observing results.

## Prediction formulation

One supervised row represents a candidate
`(graph, noise condition, QAOA configuration)`. The regression target is mean
expected approximation ratio, aggregated over the two run seeds. The model
scores every feasible candidate and recommends the candidate with the largest
predicted ratio.

Only pre-selection information is used:

- graph family, size, edge count, and density;
- deterministic structural descriptors reconstructed from the frozen graph
  seed (degree statistics, triangles, clustering, transitivity, connectivity,
  and spectral radius);
- known noise condition; and
- declared candidate descriptors (depth, optimizer, shots, and configuration
  ID).

The exact optimum, QAOA outcomes, runtime, and post-execution measurements for
the target graph are not features.

## Resource budgets

Budgets are feasibility constraints on shots per circuit:

- B256 permits configurations with at most 256 shots per circuit.
- B512 permits configurations with at most 512 shots per circuit.

This definition is used because noisy N1/N2 rows evaluate parameters
transferred from N0 rather than independently optimizing them. Comparing total
executed shots across N0 and N1/N2 as though they represented identical work
would therefore be misleading.

## Model and feature selection

Ridge, random forest, extremely randomized trees (ExtraTrees), and histogram
gradient boosting are fit on the 62 training graphs. The primary validation
criterion is mean selection regret over all validation
`(graph, noise, budget)` cases:

`regret = feasible oracle quality - selected quality`.

Row-level RMSE is secondary. Feature ablations are also evaluated only on the
validation partition. ExtraTrees with the full structural feature set is then
refit on train plus validation. The frozen test partition is used only for the
final evaluation.

## Baselines and metrics

The principal fixed baseline chooses one configuration for each
`(noise condition, budget)` using training graphs only. An analytic uniform
random expectation and the per-case feasible oracle provide lower and upper
references.

Reported metrics include mean selected quality, absolute and relative gain
over fixed, mean regret, exact oracle selection accuracy, top-3 oracle
coverage, and row-level MAE/RMSE/R².

Paired uncertainty is calculated at the graph level: bootstrap confidence
intervals resample graphs, and exact two-sided sign-flip tests operate on each
graph's mean ML-minus-fixed difference. Holm adjustment is reported across the
six budget-by-noise comparisons.

## Interpretability

Selector-aware grouped permutation importance is computed on validation data,
not test data. Features are permuted coherently at their natural unit (graph,
noise label, or configuration), and importance is the resulting increase in
selection regret. This answers how each feature group affects the decision
quality rather than only its row-wise prediction error.

## Reproduction

```bash
python scripts/finalize_ml_selector.py
```

Outputs are written to `data/ml/final_selector/`, including the analysis
manifest, validation comparisons and ablations, test selections and summaries,
graph-clustered statistics, permutation importance, and paper-ready figures.
