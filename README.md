# NQComp 2027 — QAOA + Resource-Aware ML Configuration Selector

A reproducible MaxCut benchmark and machine-learning configuration-selection pipeline for QAOA under ideal and controlled noisy execution.

The project studies whether graph-aware, pre-execution machine learning can select a high-performing QAOA configuration for an unseen graph under explicit shot budgets and noise conditions.

> **Current primary snapshot:** the earlier 89-graph analysis is superseded by the completed 100-graph noisy dataset and the deterministic 70/15/15 graph-level split described below. Historical artifacts may remain in the repository for provenance, but the current paper should use the 100-graph snapshot only.

## 1. Research question

For a previously unseen MaxCut graph, can a frozen ML selector choose a high-performing QAOA configuration without using the target graph's measured QAOA outcomes as features, while respecting declared shot-feasibility budgets and known noise conditions?

The selector operates on graph structure and configuration/context descriptors available before execution. Target-graph QAOA outcomes are excluded from the feature set.

## 2. Current frozen raw experiment

### Graph population

- 100 deterministic graph instances.
- Families: Erdős–Rényi and random-regular.
- Node counts: 10, 12, 15, 18, 20.
- Erdős–Rényi probability: 0.35.
- Random-regular degree: 4.
- Persistent graph IDs: `G0001` … `G0100`.
- Two run seeds per graph/configuration/condition.

The current raw snapshot has **complete N0, N1, and N2 coverage for all 100 graphs**. Therefore the primary ML population is the full 100-graph set; no graph is discarded for missing noisy coverage.

### QAOA configuration space

12 configurations are evaluated:

| Parameter | Values |
|---|---|
| QAOA depth | 1, 2, 3 |
| Optimizer | COBYLA, SPSA |
| Shots | 256, 512 |
| Optimizer-evaluation limit | 80 |
| Circuit-execution limit | 200 |

The canonical configuration IDs are C01–C12. Each graph/configuration/noise condition has two run seeds.

### Noise

- **N0:** ideal simulation.
- **N1:** controlled moderate noise.
- **N2:** controlled higher noise.

For N1/N2, the corresponding N0 optimized parameters are reused rather than independently optimizing every noisy condition. This isolates configuration robustness under the controlled noise settings.

### Current raw-data accounting

```text
Graphs:              100
Configurations:       12
Noise conditions:      3
Run seeds:              2
Raw observations:   7200
Aggregated ML rows:  3600
```

The raw QAOA result files remain immutable experimental artifacts. ML transformations are written separately under `data/ml/`.

### Primary quality metric

The primary quality target is the mean expected approximation ratio, aggregated over the two run seeds. Best-sampled performance is retained as a secondary descriptive quantity.

For these small graphs, exact MaxCut is available as the reference optimum.

## 3. Leakage-safe ML dataset and split

The ML dataset is derived from the frozen raw result snapshot. The common dataset contains **100 graphs**, with complete N0/N1/N2 coverage and all 12 configurations.

The split is deterministic and performed **at graph level**, using seed 2027:

| Split | Graphs | Rows |
|---|---:|---:|
| Train | 70 | 2520 |
| Validation | 15 | 540 |
| Test | 15 | 540 |
| **Total** | **100** | **3600** |

Each graph appears in exactly one partition. There is zero graph overlap across train, validation, and test. Because every graph has the same three noise conditions and twelve configurations, each split contains balanced N0/N1/N2 coverage.

The final selector is chosen using validation data only, then refit on train + validation. The test set is evaluated once after the model and feature set are frozen.

## 4. Current final ML selector

The current final selector uses **Extra Trees** with pre-execution graph-structure and configuration/context features, including graph size, density, degree statistics, clustering/connectivity descriptors, spectral descriptors, QAOA depth, optimizer, shots, configuration ID, graph family, and noise condition.

Model selection is based on **mean validation selection regret**, aligned with the actual configuration-selection task. Row-level MAE/RMSE/R² are secondary prediction diagnostics.

The current final test evaluation contains **90 graph × noise × budget cases** (15 unseen graphs × 3 noise conditions × 2 shot budgets).

### Current frozen test summary

| Method | Mean expected approximation ratio |
|---|---:|
| Uniform-random expectation | 0.694513 |
| Training-only fixed baseline | 0.707455 |
| **ML selector (Extra Trees)** | **0.711856** |
| Feasible oracle | 0.730775 |

Additional current test metrics:

```text
ML − fixed                     +0.004401
Relative gain                   +0.622148%
Mean feasible-oracle regret     0.018918
Oracle selection accuracy       22.2222%
Oracle top-3 accuracy           61.1111%
```

These are empirical results on the frozen 15-graph test partition. They should not be presented as universal superiority claims.

## 5. Baselines and robustness analyses

The study retains multiple reference strategies and diagnostic analyses:

1. **Training-only fixed configuration baseline:** selected without using test outcomes.
2. **Random configuration baseline:** seeded random selection from feasible candidates.
3. **Feasible oracle:** best measured feasible test configuration per graph/noise/budget case, used only as an upper reference.
4. **Graph nearest-neighbour heuristic:** training-only structural similarity baseline.
5. **Configuration-ID ablation:** evaluates the contribution of explicit configuration identity.
6. **Model comparison:** Ridge, Random Forest, Extra Trees, and Histogram Gradient Boosting.
7. **Feature ablations:** validation-only removal of feature groups.
8. **Selector-aware permutation importance:** validation-only interpretability analysis.
9. **Graph-clustered uncertainty/statistics:** accounts for repeated noise/budget observations belonging to the same graph.

The older global-C06 paired analysis and the earlier 89-graph selector are retained only as historical/provenance artifacts. They must not be mixed with the current 100-graph primary results in the paper.

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
- SHA-256 provenance for derived ML inputs;
- bootstrap/permutation settings;
- paper-ready CSV tables and vector/raster figures.

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
├── validate_ml_data.py
└── generate_paper_outputs.py

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

## 8. Reproducing the current analysis

From the repository root:

```bash
python scripts/validate_ml_data.py
python scripts/build_ml_dataset.py
python scripts/create_ml_splits.py
python scripts/finalize_ml_selector.py
python scripts/train_baselines.py
python scripts/statistical_analysis.py
python scripts/analyze_ml_vs_fixed_paired.py
python scripts/graph_heuristic_baseline.py
python scripts/generate_paper_outputs.py
```

The analysis scripts do not modify the canonical raw QAOA CSVs.

## 9. Research status

The **100-graph experiment and current ML selector analysis are the primary paper snapshot**. The paper should use only this current snapshot for headline results.

The paper should report the complete methodology, the 100-graph common population, the 70/15/15 graph-level split, the 12-configuration search space, N0/N1/N2 conditions, shot budgets, Extra Trees selection criterion, non-ML baselines, ablations, uncertainty/statistical analyses, and limitations.

The observed ML gain is modest in aggregate. The contribution should therefore be framed around the reproducible **resource-aware, noise-aware, graph-level, leakage-controlled configuration-selection framework and its empirical evaluation**, rather than an unsupported claim of universal performance superiority.

## 10. Repository

GitHub: https://github.com/Sanjayram3269/QAOA

**NQComp 2027 — QAOA benchmarking + controlled-noise evaluation + resource-aware ML configuration selection**
