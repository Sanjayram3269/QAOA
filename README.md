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
│   ├── graph_generation.py
│   ├── maxcut.py
│   ├── noise.py
│   ├── qaoa.py
│   └── evaluation.py
├── scripts/
│   ├── run_experiment.py
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

## Frozen quantum-to-ML contract

Each successful raw row records actual circuit executions and total executed shots. The primary quality is expected approximation ratio; best sampled ratio is secondary. B256/B512 cases are derived from the same raw evaluations, so resource conditioning does not create duplicate QAOA runs.

See `DATA_SCHEMA.md` for the complete field list, aggregation, utility, leakage, and split rules.
