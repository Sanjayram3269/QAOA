from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "ml" / "final_selector"
OUT = ROOT / "data" / "ml" / "paper_results"
FIG = OUT / "figures"
TAB = OUT / "tables"

FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD FROZEN RESULTS
# ============================================================

summary = pd.read_csv(SRC / "test_summary.csv")
models = pd.read_csv(SRC / "validation_model_comparison.csv")
ablations = pd.read_csv(SRC / "validation_feature_ablations.csv")
importance = pd.read_csv(SRC / "validation_permutation_importance.csv")
stats = pd.read_csv(SRC / "test_graph_clustered_statistics.csv")
selected = pd.read_csv(SRC / "test_selected_configurations.csv")
fixed = pd.read_csv(SRC / "fixed_baseline_choices.csv")
features = pd.read_csv(SRC / "enhanced_graph_features.csv")

# ============================================================
# HELPERS
# ============================================================

def save_table(df, name):
    df.to_csv(TAB / f"{name}.csv", index=False)
    df.to_latex(TAB / f"{name}.tex", index=False, float_format="%.4f")

def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)

# ============================================================
# TABLES
# ============================================================

# T1 — Main result
main = summary[summary["scope"] == "ALL"].copy()
save_table(main, "T1_main_results")

# T2 — Budget
budget = summary[summary["scope"].isin(["B256", "B512"])].copy()
save_table(budget, "T2_budget_results")

# T3 — Noise
noise = summary[summary["scope"].isin(["N0", "N1", "N2"])].copy()
save_table(noise, "T3_noise_results")

# T4 — Budget × noise
budget_noise = summary[
    summary["scope"].str.match(r"^B(256|512)_N[012]$")
].copy()
save_table(budget_noise, "T4_budget_noise_results")

# T5 — Model comparison
save_table(models, "T5_validation_model_comparison")

# T6 — Feature ablation
save_table(ablations, "T6_feature_ablation")

# T7 — Permutation importance
save_table(importance, "T7_permutation_importance")

# T8 — Statistical analysis
save_table(stats, "T8_statistical_tests")

# T9 — Selected configurations
save_table(selected, "T9_selected_configurations")

# T10 — Fixed baselines
save_table(fixed, "T10_fixed_baselines")

# T11 — Graph features
save_table(features, "T11_graph_features")

print("Tables generated.")

# ============================================================
# FIGURE 1 — MAIN METHOD COMPARISON
# ============================================================

r = main.iloc[0]

labels = ["Random", "Fixed", "ML", "Oracle"]
values = [
    r["random_expected_mean"],
    r["fixed_mean"],
    r["ml_mean"],
    r["oracle_mean"],
]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(labels, values)
ax.set_ylabel("Mean expected approximation ratio")
ax.set_title("Frozen test-set performance comparison")
ax.set_ylim(min(values) - 0.03, max(values) + 0.03)

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        value + 0.002,
        f"{value:.4f}",
        ha="center",
        va="bottom",
    )

savefig(fig, "F1_main_method_comparison")

# ============================================================
# FIGURE 2 — BUDGET COMPARISON
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

x = np.arange(len(budget))
width = 0.22

ax.bar(x - width, budget["random_expected_mean"], width, label="Random")
ax.bar(x, budget["fixed_mean"], width, label="Fixed")
ax.bar(x + width, budget["ml_mean"], width, label="ML")
ax.plot(x, budget["oracle_mean"], marker="o", linewidth=2, label="Oracle")

ax.set_xticks(x)
ax.set_xticklabels(budget["scope"])
ax.set_xlabel("Shot budget")
ax.set_ylabel("Mean expected approximation ratio")
ax.set_title("Performance across shot budgets")
ax.legend()

savefig(fig, "F2_budget_comparison")

# ============================================================
# FIGURE 3 — NOISE CONDITIONS
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

x = np.arange(len(noise))
width = 0.22

ax.bar(x - width, noise["random_expected_mean"], width, label="Random")
ax.bar(x, noise["fixed_mean"], width, label="Fixed")
ax.bar(x + width, noise["ml_mean"], width, label="ML")
ax.plot(x, noise["oracle_mean"], marker="o", linewidth=2, label="Oracle")

ax.set_xticks(x)
ax.set_xticklabels(noise["scope"])
ax.set_xlabel("Noise condition")
ax.set_ylabel("Mean expected approximation ratio")
ax.set_title("Performance under increasing noise")
ax.legend()

savefig(fig, "F3_noise_comparison")

# ============================================================
# FIGURE 4 — VALIDATION MODEL COMPARISON
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

m = models.sort_values("rmse")

ax.barh(m["model"], m["rmse"])
ax.set_xlabel("Validation RMSE")
ax.set_title("Validation model comparison")

savefig(fig, "F4_validation_model_comparison")

# ============================================================
# FIGURE 5 — FEATURE ABLATION
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

a = ablations.sort_values("mean_regret")

ax.barh(a["ablation"], a["mean_regret"])
ax.set_xlabel("Mean validation selection regret")
ax.set_title("Feature ablation study")

savefig(fig, "F5_feature_ablation")

# ============================================================
# FIGURE 6 — PERMUTATION IMPORTANCE
# ============================================================

imp = importance.sort_values("mean_regret_increase").tail(12)

fig, ax = plt.subplots(figsize=(8, 6))

ax.barh(
    imp["feature_group"],
    imp["mean_regret_increase"],
    xerr=imp["std_regret_increase"],
)

ax.set_xlabel("Increase in validation selection regret")
ax.set_title("Grouped selector-aware permutation importance")

savefig(fig, "F6_permutation_importance")

# ============================================================
# FIGURE 7 — REGRET DISTRIBUTION
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

ax.hist(selected["regret"], bins=15)
ax.axvline(selected["regret"].mean(), linestyle="--", linewidth=2)

ax.set_xlabel("Selection regret")
ax.set_ylabel("Number of cases")
ax.set_title("Distribution of ML selector regret")

savefig(fig, "F7_regret_distribution")

# ============================================================
# FIGURE 8 — ML VS ORACLE
# ============================================================

graph_case = (
    selected.groupby("graph_id")
    .agg(
        ml=("selected_quality", "mean"),
        oracle=("oracle_quality", "mean"),
    )
    .reset_index()
)

fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(graph_case["oracle"], graph_case["ml"])

lo = min(graph_case["oracle"].min(), graph_case["ml"].min())
hi = max(graph_case["oracle"].max(), graph_case["ml"].max())

ax.plot([lo, hi], [lo, hi], linestyle="--")

ax.set_xlabel("Oracle mean approximation ratio")
ax.set_ylabel("ML-selected mean approximation ratio")
ax.set_title("ML selector versus oracle across test graphs")

savefig(fig, "F8_ml_vs_oracle")

# ============================================================
# FIGURE 9 — CONFIGURATION SELECTION FREQUENCY
# ============================================================

freq = (
    selected["selected_config_id"]
    .value_counts()
    .sort_index()
)

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(freq.index, freq.values)

ax.set_xlabel("Selected configuration")
ax.set_ylabel("Number of test decisions")
ax.set_title("ML configuration-selection frequency")

savefig(fig, "F9_configuration_selection_frequency")

# ============================================================
# FIGURE 10 — ML VS FIXED BY NOISE
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

gain = noise["ml_minus_fixed"].astype(float)

ax.bar(noise["scope"], gain)
ax.axhline(0, linestyle="--", linewidth=1)

ax.set_xlabel("Noise condition")
ax.set_ylabel("ML − fixed approximation ratio")
ax.set_title("ML improvement relative to fixed baseline")

savefig(fig, "F10_ml_gain_over_fixed")

# ============================================================
# PAPER METADATA
# ============================================================

metadata = pd.DataFrame([
    ["Test graphs", int(selected["graph_id"].nunique())],
    ["Test decisions", len(selected)],
    ["Noise conditions", 3],
    ["Shot budgets", 2],
    ["Selected model", "ExtraTrees"],
    ["Model-selection partition", "Validation only"],
    ["Final fitting partition", "Train + validation"],
    ["Random seed", 2027],
])

metadata.columns = ["Item", "Value"]

save_table(metadata, "T12_experimental_metadata")

# ============================================================
# MANIFEST
# ============================================================

with open(OUT / "PAPER_OUTPUT_MANIFEST.txt", "w", encoding="utf-8") as f:
    f.write("NQComp 2027 — Paper Output Bundle\n")
    f.write("=" * 50 + "\n\n")
    f.write("Source: data/ml/final_selector/\n")
    f.write("All outputs are derived from frozen analysis results.\n")
    f.write("No model retraining or test-set tuning performed.\n\n")
    f.write("TABLES:\n")
    for p in sorted(TAB.glob("*")):
        f.write(f"  {p.name}\n")

    f.write("\nFIGURES:\n")
    for p in sorted(FIG.glob("*")):
        f.write(f"  {p.name}\n")

print("=" * 70)
print("PAPER OUTPUT GENERATION COMPLETE")
print("=" * 70)
print(f"Tables : {TAB}")
print(f"Figures: {FIG}")
print()
print("Generated publication bundle successfully.")