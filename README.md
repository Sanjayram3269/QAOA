# ML-Guided Resource- and Noise-Aware QAOA Configuration for Combinatorial Optimization

NQComp-2027 research project — quantum-side implementation.

## Research scope

We study whether ML can select an effective QAOA configuration for an unseen Max-Cut graph under a known execution condition.

Core configuration space:
- QAOA depth: `p ∈ {1,2,3}`
- Classical optimizer: `COBYLA`, `SPSA`
- Shots: `256`, `512`
- Total candidates: `12`
- Conditions: `N0` ideal, `N1` controlled moderate noise, `N2` controlled higher noise
- Graph families: Erdős–Rényi and Random Regular
- Graph sizes: approximately 10–20 vertices

## Repository structure

```text
QAOA/
├── config/
│   └── experiment.yaml
├── src/
│   ├── configurations.py
│   ├── graph_generation.py
│   ├── maxcut.py
│   ├── noise.py
│   ├── qaoa.py
│   └── evaluation.py
├── scripts/
│   └── smoke_test.py
├── tests/
│   └── test_quantum_foundation.py
├── DATA_SCHEMA.md
├── DECISIONS.md
├── requirements.txt
└── README.md
```

## Ownership

Sanjay owns graph generation, exact Max-Cut, QAOA, configuration registry, noise, and quantum experiment orchestration. Neha owns feature extraction, utility/oracle labels, RF/XGBoost, baselines, and ML analysis.

## Local setup

```bash
git clone https://github.com/Sanjayram3269/QAOA.git
cd QAOA
git checkout feature/sanjay-quantum-foundation
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

## First validation

Run unit tests:

```bash
pytest -q
```

Run the end-to-end smoke test:

```bash
python scripts/smoke_test.py
```

Do **not** start large experiments until the smoke test and unit tests pass locally.

## Collaboration

`main` is the stable branch. Quantum and ML work should remain on separate feature branches. The shared interface is documented in `DATA_SCHEMA.md`; methodology changes are recorded in `DECISIONS.md`.

## Status

Phase 0 — repository and quantum foundation setup.
