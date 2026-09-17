# ML-Guided Resource- and Noise-Aware QAOA Configuration for Combinatorial Optimization

NQComp-2027 research project.

## Scope

This repository contains the quantum-side implementation for a study that evaluates whether machine learning can select an effective QAOA configuration for unseen Max-Cut graph instances under different execution conditions.

### Core problem
- Problem: Max-Cut
- Graph families: Erdős–Rényi and Random Regular
- Graph size: approximately 10–20 vertices
- QAOA depth: p = 1, 2, 3
- Optimizers: COBYLA, SPSA
- Shots: 256, 512
- Execution conditions: ideal plus two controlled simulated-noise conditions

## Repository ownership

- `src/graph_generation.py`, `src/maxcut.py`, `src/qaoa.py`, `src/noise.py`, `src/configurations.py`, and quantum experiment orchestration are owned by Sanjay.
- ML feature extraction, model training, baselines, and statistical analysis are maintained in the ML branch by Neha.

## Collaboration rule

`main` is the stable branch. Quantum and ML work should be developed on separate feature branches and integrated only through the agreed data contract.

## Status

Phase 0 — repository and quantum foundation setup.
