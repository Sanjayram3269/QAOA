# NQComp Quantum Intelligence Dashboard

A premium, research-facing dashboard for the frozen `nqcomp-2027-qaoa-ml-selector` experiment.

## What it does

The dashboard is intentionally an **artifact viewer and analysis surface**, not a second experiment engine. It reads committed CSV/JSON/YAML artifacts from the repository and exposes:

- Overview and aggregate ML / fixed / random / oracle comparison
- Budget analysis (B256 / B512)
- Noise analysis (N0 / N1 / N2)
- Budget × noise matrix
- Held-out graph explorer
- Per-case selector decision traces
- Canonical 12-configuration lab
- ML selector / Extra Trees validation
- Permutation-importance explainability
- Feature ablation
- Statistical evidence with confidence intervals
- Reproducibility and experiment contract
- Methodology / reviewer-oriented explanation

## Run locally

From the repository root:

```bash
cd dashboard
npm install
npm run dev
```

The Vite server uses the repository root so the dashboard can read the authoritative research artifacts without duplicating them. Open:

```text
http://localhost:5173/dashboard/
```

Do not open `dashboard/index.html` with `file://`; the browser must receive the app through Vite so the CSV artifacts can be fetched correctly.

## Production build

```bash
cd dashboard
npm run build
npm run preview
```

## Tests

```bash
cd dashboard
npm test
```

The test command performs JavaScript syntax checking and validates that the dashboard's source artifacts exist and contain the frozen values / row contracts expected by the UI.

CI also runs a production Vite build through `.github/workflows/dashboard.yml`.

## Data contract

The UI reads the authoritative frozen artifacts under:

- `data/ml/paper_results/`
- `data/graphs/master_graph_manifest.csv`
- `config/experiment.yaml`

The most important files are `01_main_results.csv`, `02_budget_results.csv`, `03_noise_results.csv`, `04_budget_noise_results.csv`, `05_model_validation_comparison.csv`, `06_feature_ablation.csv`, `07_permutation_importance.csv`, `08_statistical_tests.csv`, `09_test_selected_configurations.csv`, `10_fixed_baseline_choices.csv`, and `11_test_graph_features.csv`.

## Design principles

1. **No invented metrics.** Cards and charts are driven from committed artifacts.
2. **No silent recomputation.** The UI does not rerun QAOA or retrain the selector.
3. **High contrast.** Critical values are never rendered as near-white text on white cards.
4. **Reviewer friendly.** Every high-level number has a route to the underlying evidence.
5. **Responsive.** The sidebar collapses on smaller screens and charts resize with the viewport.
6. **Research-safe language.** The dashboard distinguishes descriptive results, uncertainty, and reference/oracle quantities.
