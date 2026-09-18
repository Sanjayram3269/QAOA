# Methodology Decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-09-17 | Max-Cut is the sole core optimization problem. | Keeps the 21-day implementation scope focused. |
| 2026-09-17 | Core configuration space is p={1,2,3} × {COBYLA,SPSA} × {256,512 shots}. | Gives 12 candidate configurations and explicit heterogeneity. |
| 2026-09-17 | Execution conditions are N0 ideal, N1 controlled moderate noise, N2 controlled higher noise. | Enables controlled robustness analysis. |
| 2026-09-17 | Exact Max-Cut is used for graphs up to 20 nodes. | Provides an oracle optimum for approximation-ratio evaluation. |
| 2026-09-17 | Graph-level train/validation/test split is mandatory. | Prevents graph-instance leakage across configurations and noise rows. |
| 2026-09-17 | Raw quantum outputs are immutable after generation. | Preserves reproducibility and clean ML handoff. |
| 2026-09-17 | Expected cut and best sampled cut are both stored. | Allows the final analysis to choose and justify a primary quality definition without losing information. |
| 2026-09-17 | RF/XGBoost are the primary ML models. | Strong tabular baselines with manageable implementation time. |

Any change to configuration space, noise parameters, utility definition, normalization, split strategy, or primary metric must be recorded here before final experiments.
