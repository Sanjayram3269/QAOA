# ML-Guided Resource- and Noise-Aware QAOA Configuration for Combinatorial Optimization

NQComp-2027 research implementation.

## Research scope

The project studies whether machine learning can select an effective QAOA configuration for an unseen Max-Cut graph under known noise and resource conditions.

Core configuration space:

- Depth: p in {1, 2, 3}
- Optimizer: COBYLA or SPSA
- Shots per circuit: 256 or 512
- Twelve configurations: C01-C12
- Noise: N0 ideal, N1 moderate, N2 higher simulated noise
- Resource budgets: B256 and B512
- Graph families: Erdos-Renyi and 4-regular random graphs
- Graph sizes: 10, 12, 15, 18, and 20 vertices
- Feasibility repetitions: paired seeds 20000, 20001, and 20002
- Maximum resource use: 200 measured circuit executions per candidate

## Branch workflow

- `main` is stable.
- Sanjay develops the quantum pipeline on quantum feature branches.
- Neha develops features, utility, datasets, models, baselines, and analysis on ML feature branches.
- Changes enter `main` through reviewed pull requests.
- Shared interfaces are defined in `DATA_SCHEMA.md`.
- Methodology decisions are recorded in `DECISIONS.md`.

## Repository structure

```text
QAOA/
├── config/
│   └── experiment.yaml
├── data/
│   ├── graphs/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   ├── configurations.py
│   ├── evaluation.py
│   ├── features.py
│   ├── graph_generation.py
│   ├── maxcut.py
│   ├── ml_dataset.py
│   ├── ml_selector.py
│   ├── noise.py
│   ├── qaoa.py
│   ├── splitting.py
│   └── validation.py
├── scripts/
│   ├── run_experiment.py
│   ├── run_ml_pipeline.py
│   └── smoke_test.py
├── tests/
├── DATA_SCHEMA.md
├── DECISIONS.md
├── requirements.txt
└── README.md
```

## Local setup

```bash
git clone https://github.com/Sanjayram3269/QAOA.git
cd QAOA
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Validation

Run the unit tests:

```bash
pytest -q
```

Run the smallest QAOA smoke test:

```bash
python scripts/smoke_test.py
```

Run the one-graph, all-configuration raw-record smoke experiment:

```bash
python scripts/run_experiment.py --smoke
```

The latter writes `data/raw/batch_sanjay/smoke_results.csv`. Do not start the 50-graph feasibility run until all tests and both smoke paths pass and every raw row reports `run_status=success`.

## ML selector

Prepare and validate the smoke handoff:

```bash
python scripts/run_ml_pipeline.py \
  --input data/raw/batch_sanjay/smoke_results.csv \
  --output-dir data/processed/ml_selector
```

For the one-graph smoke file, the command validates the contract and builds aggregate results, oracle labels, reconstructed graph features, and the two budget-conditioned ML examples. Training is intentionally skipped because graph-level splitting is impossible with one graph.

Run the same command with `feasibility_results.csv` after the 5,400-run experiment completes. It will then:

1. split the 50 unique graphs 70/15/15 with seed 2027;
2. compare a Random Forest with multinomial logistic regression on validation utility regret;
3. mask budget-infeasible configurations at prediction time;
4. refit the selected model on train plus validation graphs;
5. evaluate once on the untouched test graphs;
6. compare against the condition-aware global-best baseline; and
7. save predictions, metrics, graph assignments, and the fitted model.

Graph identifiers and generation seeds are retained for traceability but are not model inputs. No QAOA outcome, exact optimum, oracle value, resource measurement, or configuration field enters the selector.

## Frozen quantum-to-ML contract

Each successful raw row records actual circuit executions and total executed shots. The primary quality is expected approximation ratio; best sampled ratio is secondary. B256/B512 cases are derived from the same raw evaluations, so resource conditioning does not create duplicate QAOA runs.

See `DATA_SCHEMA.md` for the complete field list, aggregation, utility, leakage, and split rules.
