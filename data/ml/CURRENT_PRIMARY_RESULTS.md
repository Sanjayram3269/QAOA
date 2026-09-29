# Current Primary ML Results Snapshot

**Status:** primary paper snapshot
**Date:** 2026-09-29
**Split seed:** 2027

This file records the current primary experiment as reported by the local
frozen analysis run. It supersedes the earlier 89-graph result snapshot for
paper writing.

## Dataset

| Quantity | Current value |
|---|---:|
| Graphs | 100 |
| Configurations | 12 |
| Noise conditions | 3 (N0/N1/N2) |
| Run seeds | 2 |
| Raw observations | 7200 |
| Seed-aggregated ML rows | 3600 |

Coverage is complete for all 100 graphs under N0, N1, and N2.

## Graph-level split

| Partition | Graphs | Rows |
|---|---:|---:|
| Train | 70 | 2520 |
| Validation | 15 | 540 |
| Test | 15 | 540 |

There is zero graph overlap between partitions.

## Model selection

The current final selector is **ExtraTrees** with the full pre-execution graph
and configuration feature set. Model selection is performed on validation data
using mean selection regret. The final model is refit on train + validation and
evaluated once on the held-out test graphs.

## Frozen test selector result

The test set contains:

```text
15 graphs × 3 noise conditions × 2 budgets = 90 cases
```

| Method | Mean expected approximation ratio |
|---|---:|
| Uniform-random expectation | 0.694513 |
| Training-only fixed baseline | 0.707455 |
| ML selector (ExtraTrees) | 0.711856 |
| Feasible oracle | 0.730775 |

```text
ML − fixed:                 +0.004401
Relative gain:              +0.622148%
Mean oracle regret:          0.018918
Oracle selection accuracy:  22.2222%
Oracle top-3 accuracy:       61.1111%
```

## Global C06 baseline reference

The separate non-ML baseline script selects C06 from training data only:

```text
C06 training mean:          0.704136
C06 test mean (ALL):        0.704042
```

This global-C06 analysis must not be conflated with the task-aligned fixed
baseline used by the resource-aware final selector, which is selected per
noise/budget case from training data only.

## Paper-use rule

Use the **100-graph / 70-15-15 / ExtraTrees / 90-case** results above as the
primary paper snapshot. Do not mix the earlier 89-graph / 62-13-14 results
into headline tables or conclusions.

The full generated row-level CSVs and publication figures remain the canonical
machine-readable artifacts produced by the local pipeline; this file is a
human-readable checkpoint of the current headline results.
