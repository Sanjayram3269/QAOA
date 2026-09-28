# NQComp 2027 — QAOA + Resource-Aware ML Configuration Selector

A reproducible MaxCut benchmark and machine-learning configuration-selection pipeline for QAOA under ideal and controlled noisy execution.

The project studies whether graph-aware, pre-execution machine learning can select a high-performing QAOA configuration for an unseen graph, subject to shot-feasibility budgets and noise conditions.

## 1. Research question

For a previously unseen MaxCut graph, can a frozen ML selector choose a QAOA configuration that performs at least as well as non-ML baselines without using the target graph's measured QAOA outcomes as features?

The selector operates on graph structure and configuration/context descriptors available before execution. Target-graph QAOA outcomes are excluded from the feature set.

## 2. Frozen raw experiment

### Graph population

- 100 deterministic graph instances were generated.
- Families: Erdős–Rényi and random-regular.
- Node counts: 10, 12, 15, 18, 20.
- Erdős–Rényi probability: 0.35.
- Random-regular degree: 4.
- Persistent graph IDs: `G0001` … `G0100`.
- Graph-generation seed starts at 10000.

The raw benchmark contains complete N0 coverage for all 100 generated graphs. N1/N2 coverage is complete through G0089. For the primary ML analysis, only graphs with complete N0/N1/N2 coverage and all 12 configurations are admitted, producing a common population of **89 graphs**.

### QAOA configuration space

12 configurations are evaluated:

| Parameter | Values |
|---|---|
| QAOA depth | 1, 2, 3 |
| Optimizer | COBYLA, SPSA |
| Shots | 256, 512 |
| Optimizer-evaluation limit | 80 |
| Circuit-execution limit | 200 |

The canonical configuration IDs are C01–C12. The raw experiment uses two run seeds per graph/configuration/condition: 1000 and 1001.

### Noise

- **N0:** ideal simulation.
- **N1:** controlled moderate depolarizing/readout noise.
- **N2:** controlled higher depolarizing/readout noise.

For N1/N2, the experiment reuses the corresponding N0 optimized parameters rather than independently optimizing every noisy condition. This isolates configuration robustness under noise.

### Primary quality metric

The primary quality target is the mean expected approximation ratio, retained at the seed-aggregation stage. Best-sampled performance is treated as a secondary quantity.

For supported small graphs, exact MaxCut is computed as the reference optimum.

## 3. Frozen ML dataset and split

The ML dataset is derived from the immutable raw result snapshot. The common dataset contains 89 graphs with complete N0/N1/N2 and 12-configuration coverage.

The split is performed **at graph level**, using seed 2027:

| Split | Graphs |
|---|---:|
| Train | 62 |
| Validation | 13 |
| Test | 14 |
| Total | 89 |

There is zero graph overlap between train, validation, and test partitions.

The final selector is selected using validation data only, then refit on train + validation. The test set is evaluated once after the model and feature set are frozen.

The frozen ML input is `data/ml/splits/ml_performance_common.csv`, whose SHA-256 is recorded in `data/ml/final_selector/analysis_manifest.json`.

## 4. Final ML selector

The final selector uses **Extra Trees** with pre-execution graph-structure and configuration/context features, including graph size, density, degree statistics, clustering/connectivity descriptors, spectral descriptors, QAOA depth, optimizer, shots, configuration ID, graph family, and noise condition.

Model selection is based on **mean validation selection regret**, the metric aligned with the actual configuration-selection task. Row-level prediction MAE/RMSE/R² are reported as secondary diagnostics.

Final analysis provenance is recorded in:

```text
data/ml/final_selector/analysis_manifest.json
```

Key frozen final-test metrics:

```text
ML mean                         0.702228
Fixed baseline mean             0.700960
Random expected mean            0.689134
Oracle mean                     0.724351
ML − fixed                      +0.001269
Relative gain                   +0.181%
Mean regret                     0.022123
Oracle selection accuracy       17.86%
Oracle top-3 accuracy            58.33%
```

The small aggregate gain is reported as an empirical result rather than as a claim of universal superiority. Performance is also reported separately by budget and noise condition.

## 5. Baselines and robustness analyses

The study includes multiple reference strategies:

1. **Training-only fixed configuration baseline.** The fixed choice is determined without using test outcomes.
2. **Random configuration baseline.** Seeded random selection from feasible configurations.
3. **Oracle reference.** Best measured test configuration per graph/condition, used only as an unattainable upper reference.
4. **Graph nearest-neighbour heuristic.** A training-only, k=5 standardized graph-feature heuristic using number of nodes, edges, and density.
5. **Configuration-ID ablation.** Tests the effect of including configuration identity among the selector features.
6. **Model and feature ablations.** Validation-only comparisons and permutation importance are retained in the repository.
7. **Paired ML-vs-fixed statistical analysis.** The frozen global C06 baseline is compared on exactly the same 42 graph/noise observations using a Wilcoxon signed-rank test.

The paired C06 analysis reports:

```text
ALL: ML mean 0.708474 vs C06 0.698668
Mean paired difference: +0.009806
Wilcoxon p-value: 0.004625
Paired Cohen's d: 0.392
```

Noise-specific results are retained in `data/ml/final_analysis/paired_ml_vs_fixed_statistics.csv`. The paper should distinguish this global-C06 paired analysis from the stronger task-aligned fixed-baseline comparison used by the final selector, where the fixed configuration is selected separately for each noise/budget case using training graphs only.

## 6. Reproducibility and leakage controls

The project preserves:

- deterministic graph IDs and seeds;
- a frozen 12-configuration registry;
- immutable raw QAOA result files;
- graph-level train/validation/test separation;
- validation-only model/feature selection;
- train + validation refitting before the final test;
- no target-graph QAOA outcomes as selector features;
- explicit N0/N1/N2 condition labels;
- training-only fixed baselines;
- oracle results only as reference upper bounds;
- SHA-256 provenance for the final ML input;
- bootstrap and permutation-importance settings;
- paper-ready CSV tables and figures.

## 7. Key files

```text
config/experiment.yaml
DECISIONS.md
ML_SELECTOR_METHOD.md
DATA_SCHEMA.md

scripts/
├── run_experiment.py
├── build_ml_dataset.py
├── create_ml_splits.py
├── train_ml_selector.py
├── compare_ml_models.py
├── finalize_ml_selector.py
├── train_baselines.py
├── statistical_analysis.py
├── analyze_ml_vs_fixed_paired.py
├── graph_heuristic_baseline.py
└── validate_ml_data.py

data/ml/
├── dataset_manifest.json
├── splits/
├── baselines/
├── models/
├── model_comparison/
├── final_selector/
├── final_analysis/
├── paper_results/
└── upgrades/
    ├── config_id_ablation/
    └── graph_heuristic/
```

## 8. Reproducing the final selector analysis

From the repository root:

```bash
python scripts/validate_ml_data.py
python scripts/finalize_ml_selector.py
python scripts/statistical_analysis.py
python scripts/analyze_ml_vs_fixed_paired.py
python scripts/graph_heuristic_baseline.py
```

The final selector analysis does not modify the canonical raw QAOA CSVs.

## 9. Research status

The experimental evidence is now frozen for paper writing. Numerical results should not be regenerated or tuned against the test set while drafting the paper.

The final paper should report the complete methodology, the 89-graph common population, the 62/13/14 graph-level split, the 12-configuration search space, N0/N1/N2 conditions, resource budgets, Extra Trees selection criterion, non-ML baselines, ablations, uncertainty/statistical analyses, and the observed limitations.

In particular, the paper should not overstate the aggregate ML gain: the contribution is the **resource-aware, noise-aware, graph-level, leakage-controlled configuration-selection framework and its empirical evaluation**, with the final test performance reported transparently across conditions.

## 10. Repository

GitHub: https://github.com/Sanjayram3269/QAOA

**NQComp 2027 — QAOA benchmarking + noise-aware evaluation + resource-aware ML configuration selection**
