# Methodology Decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-09-17 | Max-Cut is the sole core optimization problem. | Protects the core implementation and paper scope. |
| 2026-09-17 | Core configuration space is p={1,2,3} x {COBYLA,SPSA} x {256,512 shots}. | Provides 12 controlled candidates. |
| 2026-09-17 | Execution conditions are N0 ideal, N1 moderate simulated noise, and N2 higher simulated noise. | Enables controlled robustness analysis. |
| 2026-09-17 | Exact Max-Cut is used for graphs up to 20 nodes. | Supplies the reference optimum for approximation ratios. |
| 2026-09-17 | Graph-level train/validation/test splitting is mandatory. | Prevents graph-instance leakage. |
| 2026-09-17 | Raw quantum outputs are immutable and versioned. | Preserves reproducibility and a clean handoff. |
| 2026-09-18 | The regular-graph degree is 4. | Degree 3 is invalid for the included 15-node graphs because n x degree must be even. |
| 2026-09-18 | B256 and B512 are pre-selection maximum-shots-per-circuit budgets. | Makes resource conditioning explicit without additional QAOA runs. |
| 2026-09-18 | Each candidate is capped at 200 measured circuit executions, including final evaluation. | Makes optimizer resource use comparable and bounds cost. |
| 2026-09-18 | The primary cost cap is 102400 shots. | Equals 512 shots x 200 circuit executions. |
| 2026-09-18 | Feasibility uses paired run seeds 20000, 20001, and 20002. | Supports stable seed aggregation and paired comparisons. |
| 2026-09-18 | Mean expected approximation ratio is the primary quality metric; best sampled ratio is secondary. | Best sampled cuts often saturate the exact optimum and provide little configuration discrimination. |
| 2026-09-18 | Primary utility uses alpha=0.8; sensitivity uses 0.6, 0.7, 0.8, and 0.9. | Quality remains primary while resource cost is explicit. |
| 2026-09-18 | Utilities within 0.005 of the maximum are practically tied. | Reduces unstable oracle labels caused by negligible stochastic differences. |
| 2026-09-18 | Practical ties prefer lower total shots, lower depth, fewer shots per circuit, then lower configuration ID. | Produces deterministic, resource-conscious labels. |
| 2026-09-18 | Unique graphs are split 70/15/15 with seed 2027 before row expansion. | Prevents leakage and makes the split reproducible. |
| 2026-09-18 | The condition-aware global-best baseline is learned separately for each noise-budget condition from training data only. | Provides a strong non-graph-adaptive comparison. |

Any later change to configuration space, noise parameters, utility, normalization, resource budgets, seed policy, split strategy, baseline definitions, or primary metrics must be recorded here before affected experiments are run.
