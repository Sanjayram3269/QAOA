README.md — NQComp 2027 QAOA + ML Selector
MaxCut • QAOA benchmarking • Noise-aware evaluation • ML-based configuration selection

NQComp 2027 — QAOA + ML Selector

A reproducible experimental pipeline for benchmarking **QAOA (Quantum Approximate Optimization Algorithm)** on the **MaxCut** problem and using the resulting experimental data as the foundation for a machine-learning-based QAOA configuration selector.

The project is designed around controlled graph instances, multiple QAOA depths/optimizers/shots, ideal execution (N0), and noisy execution (N1/N2), with resumable experiment execution and CSV checkpoints.

---

1. Project Objective

The main objective is to investigate how QAOA performance changes across:

- Graph structure
- Graph size
- QAOA circuit depth
- Classical optimizer
- Number of measurement shots
- Noise condition

The experimental dataset is subsequently intended to support an **ML selector** that predicts or recommends suitable QAOA configurations for a given graph/experimental condition.

---

2. Problem

MaxCut

Given an undirected graph:

**G = (V, E)**

the MaxCut objective is to partition the vertices into two sets so that the number/weight of edges crossing the partition is maximized.

For a candidate bitstring:

**x = x₁x₂...xₙ**

the corresponding cut value is computed from the graph edges.

For small graphs, an exact classical solver is used to obtain the optimum MaxCut value. This provides a reference against which QAOA results can be evaluated.

---

3. Experimental Design

Graph families

The experiment uses:

- Erdős–Rényi random graphs
- Random regular graphs

Current graph sizes:

- 10 nodes
- 12 nodes
- 15 nodes
- 18 nodes
- 20 nodes

Graph instances are assigned persistent IDs:

`G0001, G0002, ..., G0100`

A master graph manifest is used so that experiments can be resumed without regenerating or changing graph instances.

---

4. QAOA Configuration Space

The configuration registry covers:

| Parameter | Values |
|---|---|
| QAOA depth (p) | 1, 2, 3 |
| Optimizer | COBYLA, SPSA |
| Shots | 256, 512 |
| Maximum optimizer evaluations | Project-configured limit |
| Noise condition | N0, N1, N2 |

SPSA uses the project-configured values for its hyperparameters.

Each graph/configuration/noise combination is stored as an individual experimental row.

---

5. Noise Conditions

N0 — Ideal

Ideal/noiseless simulation.

N1 — Moderate noise

The configured moderate-noise model.

N2 — Higher noise

The configured higher-noise model.

The noise experiments are stored with an explicit `noise_condition` field so N1 and N2 can be separated during analysis and ML preprocessing.

---

6. Experiment Pipeline

The high-level pipeline is:

```text
Graph Generation
      ↓
Master Graph Manifest
      ↓
Exact MaxCut Reference
      ↓
QAOA Configuration Registry
      ↓
N0 / N1 / N2 Simulation
      ↓
Checkpointed CSV Results
      ↓
Validation / Recovery
      ↓
Feature Engineering
      ↓
ML Configuration Selector
      ↓
Evaluation
```

---

7. Repository Structure

```text
QAOA/
│
├── config/
│   └── experiment configuration files
│
├── data/
│   ├── graphs/
│   │   └── master_graph_manifest.csv
│   │
│   └── results/
│       ├── n0_results.csv
│       └── noisy_results.csv
│
├── scripts/
│   ├── run_experiment.py
│   ├── smoke_test.py
│   └── supporting experiment/recovery scripts
│
├── src/
│   ├── QAOA implementation
│   ├── evaluation
│   ├── graph generation
│   ├── noise models
│   ├── feature extraction
│   └── validation/recovery utilities
│
├── tests/
│
├── DATA_SCHEMA.md
├── DECISIONS.md
├── README.md
└── requirements.txt
```

---

8. Reproducibility

Experiments use deterministic graph-generation seeds/configuration wherever applicable.

Important reproducibility components include:

- Graph manifest
- Graph IDs
- Experiment configuration
- Random seeds
- QAOA configuration registry
- Noise condition
- Checkpointed result files

The graph manifest should be generated before running the full experiment:

```bash
python -m src.graph_generation
```

The resulting manifest should be preserved together with the result CSVs.

---

9. Smoke Test

Before launching a long experiment, run the smoke test:

```bash
python scripts/smoke_test.py
```

The smoke test verifies the core experiment path without requiring the complete benchmark.

A successful smoke test should confirm that:

- Graph loading works
- QAOA execution works
- Exact evaluation works
- Results can be generated
- Required fields are produced

---

10. Running Experiments

The experiment runner supports resumable modes.

Check the available options:

```bash
python scripts/run_experiment.py --help
```

Current interface:

```text
--mode {n0,noise}
--batch BATCH
--batch-size BATCH_SIZE
--graph-ids GRAPH_IDS [GRAPH_IDS ...]
```

N0

Example:

```bash
python scripts/run_experiment.py --mode n0 --batch 1
```

Noise

Example:

```bash
python scripts/run_experiment.py --mode noise --graph-ids G0088 G0089
```

When explicit graph IDs are supplied, the runner executes only those graph instances.

This is useful for recovery of interrupted experiments.

---

11. Resumability

Long simulations are checkpointed to CSV.

The runner checks existing results and executes only missing graph/configuration/noise combinations.

This prevents already-completed evaluations from being unnecessarily recomputed after:

- Colab runtime termination
- GPU/session interruption
- Manual stopping
- Partial experiment failure

Always preserve the latest CSV checkpoint before terminating a runtime.

---

12. Current Experimental Dataset Checkpoint

At the current project checkpoint:

N0

```text
Rows:   2400
Graphs: 100
Range:  G0001 → G0100
```

N1/N2

```text
Rows:   4272
Graphs: 89
Range:  G0001 → G0089

N1: 2136 rows
N2: 2136 rows
```

Therefore, the current noisy dataset contains complete N1/N2 results through **G0089**.

Graphs **G0090–G0100** remain to be completed if the full 100-graph noisy benchmark is required.

---

13. Result Data

The main result files are:

```text
data/results/n0_results.csv
data/results/noisy_results.csv
```

Do not confuse uploaded/downloaded copies such as:

```text
n0_results (1).csv
noisy_results (2).csv
```

with the canonical repository paths.

Before analysis, validate the files:

```python
import pandas as pd

n0 = pd.read_csv("data/results/n0_results.csv")
noise = pd.read_csv("data/results/noisy_results.csv")

print("N0:", n0.shape)
print("Noise:", noise.shape)

print("N0 graphs:", n0["graph_id"].nunique())
print("Noise graphs:", noise["graph_id"].nunique())
```

---

14. Data Integrity Checks

Recommended checks before ML:

```python
assert n0["graph_id"].nunique() == 100
assert noise["graph_id"].nunique() == 89
assert set(noise["noise_condition"].unique()) == {"N1", "N2"}
```

Also check for:

- Duplicate graph/configuration rows
- Missing configuration combinations
- Missing graph IDs
- Invalid cut values
- Missing exact optimum values
- Unexpected noise labels
- NaN/Inf values

The exact expected row count should be derived from the configuration registry rather than hard-coded whenever possible.

---

15. GPU Execution

The project can use Qiskit Aer GPU simulation when the runtime provides a compatible GPU-enabled Aer installation.

A working GPU test should use:

```python
from qiskit_aer import AerSimulator

sim = AerSimulator(method="statevector", device="GPU")
print(sim.available_devices())
```

A successful environment should report GPU availability and successfully execute a small circuit.

Example verification:

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

sim = AerSimulator(device="GPU")
job = sim.run(qc, shots=100)
result = job.result()

print(result.get_counts())
```

If the runtime reports:

```text
Simulation device "GPU" is not supported on this system
```

the Aer installation/runtime is not currently configured for GPU execution, even if `nvidia-smi` shows a GPU.

---

16. Exact Solver

For small graph sizes used in this project, an exact classical MaxCut solver can be used as a reference.

The exact optimum is useful for calculating metrics such as:

- Approximation ratio
- Optimality gap
- Success against the exact optimum
- QAOA expected cut relative to the optimum

A typical approximation ratio is:

```text
Approximation Ratio =
QAOA objective / Exact MaxCut optimum
```

The exact definition used in analysis should match the project's evaluation implementation.

---

17. ML Stage

After the experimental dataset is finalized, the next stage is the ML selector.

Potential input features include:

Graph features

- Number of nodes
- Number of edges
- Graph density
- Degree statistics
- Degree variance
- Clustering-related statistics
- Other graph-structure descriptors implemented by the project

QAOA/context features

- Noise condition
- Hardware/simulation condition where applicable
- Configuration metadata

Target

The ML system can be formulated around selecting or predicting a high-performing QAOA configuration.

The exact target definition should be fixed before training and documented in the ML experiment configuration.

---

18. Recommended ML Evaluation

The ML stage should use a graph-aware train/validation/test split to reduce leakage between structurally related instances.

Evaluation should report appropriate metrics for the selected formulation, for example:

- Configuration-selection accuracy
- Top-k selection accuracy
- Predicted vs. actual performance
- Approximation ratio
- Regret relative to the best available configuration
- Performance under N0/N1/N2 separately

The baseline should be explicitly defined so the ML selector can be compared against non-ML configuration-selection strategies.

---

19. Research Reproducibility Checklist

Before final paper analysis:

```text
[ ] Master graph manifest preserved
[ ] Graph IDs fixed
[ ] Random seeds documented
[ ] Configuration registry preserved
[ ] N0 results validated
[ ] N1 results validated
[ ] N2 results validated
[ ] Duplicate rows checked
[ ] Missing combinations checked
[ ] Exact solver outputs validated
[ ] Result schema documented
[ ] ML target definition frozen
[ ] Train/validation/test split frozen
[ ] Baselines defined
[ ] Final experiment scripts committed
```

---

20. Important Project Principle

The raw experimental CSV files are research data.

Do not silently overwrite, regenerate, or modify them during analysis.

Prefer:

```text
raw results
    ↓
validated copy
    ↓
feature-engineered dataset
    ↓
ML dataset
    ↓
model outputs
```

This keeps the original experiment reproducible and makes it possible to trace every ML result back to the underlying QAOA experiment.

---

21. Project Status

Completed

- Graph-generation pipeline
- Master graph manifest
- Exact MaxCut evaluation
- QAOA execution pipeline
- N0 benchmark for 100 graphs
- N1/N2 benchmark through G0089
- Resumable experiment execution
- CSV checkpointing
- GPU-compatible execution path
- Validation/recovery utilities

Remaining

- Complete N1/N2 for G0090–G0100, if full noisy coverage is required
- Freeze and validate the final research dataset
- Feature engineering
- ML configuration-selector training
- Baseline comparison
- Ablation analysis
- Final plots/tables
- Paper-ready statistical analysis

---

22. Citation / Research Use

This repository contains the implementation and experimental artifacts for the NQComp 2027 project.

When presenting results, report the exact:

- Graph population
- Configuration space
- Noise model
- Number of completed evaluations
- Random seed policy
- Simulator/backend
- Evaluation metric
- ML split strategy

This prevents partial experimental coverage from being presented as a full benchmark.

---

23. Quick Start

```bash
git clone <repository-url>
cd QAOA

pip install -r requirements.txt

python -m src.graph_generation

python scripts/smoke_test.py

python scripts/run_experiment.py --help
```

Then run the required experiment mode and preserve the generated CSV checkpoints.

---

24. Repository

GitHub:

https://github.com/Sanjayram3269/QAOA

---

25. Authors / Project
	
**NQComp 2027**

Project focus:

**QAOA benchmarking + noise-aware evaluation + ML-based QAOA configuration selection**
