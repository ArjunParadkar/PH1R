"""
PRISM-AD Quick Visuals
======================
Run:  python3 visuals.py
Saves PNG files next to this script (or opens interactively).

Produces four simple charts from results/tables/:
  1. val_mse_by_model.png      – Val MSE per model, one bar group per dataset
  2. boundary_mse_by_model.png – Boundary MSE per model, one bar group per dataset
  3. topo_dist_by_model.png    – Topology distance per model, one bar group per dataset
  4. speedup_bar.png           – Solver speedup per model (log scale)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – remove this line to show windows instead
import matplotlib.pyplot as plt
import csv

# ── paths ────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.abspath(__file__))
TABLES   = os.path.join(ROOT, "results", "tables")
OUT_DIR  = ROOT                # save next to this file

METRICS_CSV = os.path.join(TABLES, "model_metrics_all_datasets.csv")
RUNTIME_CSV = os.path.join(TABLES, "runtime_speedup.csv")

# ── colour palette (one per model) ───────────────────────────────────────────
PALETTE = {
    "MLP":      "#4C72B0",
    "CNN":      "#DD8452",
    "FNO":      "#55A868",
    "PI-FNO":   "#C44E52",
    "TOPO-FNO": "#8172B3",
}

# ── helpers ───────────────────────────────────────────────────────────────────
def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def nice_dataset(name):
    return {
        "benchmark_operator":   "Benchmark",
        "vlasov_maxwell_gkeyll": "Vlasov-Maxwell",
        "fusion_real_lhd":      "LHD",
        "d3d_real":             "D3D",
        "fusion_real_tcv":      "TCV",
    }.get(name, name)


def bar_group_chart(datasets, models, values, ylabel, title, filename,
                    log=False, fmt="{:.2e}"):
    """datasets × models matrix → grouped bar chart."""
    n_ds  = len(datasets)
    n_mod = len(models)
    x     = np.arange(n_ds)
    width = 0.8 / n_mod

    fig, ax = plt.subplots(figsize=(max(6, n_ds * 1.8), 5))
    for i, model in enumerate(models):
        offsets = x + (i - n_mod / 2 + 0.5) * width
        vals    = [values[ds].get(model, float("nan")) for ds in datasets]
        bars    = ax.bar(offsets, vals, width * 0.9,
                         label=model, color=PALETTE.get(model, None))
        # value labels on top of bars
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * (1.02 if not log else 1.1),
                        fmt.format(v), ha="center", va="bottom",
                        fontsize=6, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([nice_dataset(d) for d in datasets], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=n_mod)
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved  {path}")


# ── load metrics ─────────────────────────────────────────────────────────────
rows    = read_csv(METRICS_CSV)
val_rows = [r for r in rows if r["Subset"] == "val"]

datasets = list(dict.fromkeys(r["Dataset"] for r in val_rows))
models   = list(dict.fromkeys(r["Model"]   for r in val_rows))

def build_matrix(field):
    m = {ds: {} for ds in datasets}
    for r in val_rows:
        m[r["Dataset"]][r["Model"]] = float(r[field])
    return m

val_mse   = build_matrix("MSE")
bnd_mse   = build_matrix("Boundary_MSE")
topo_dist = build_matrix("Topology_Distance")

# ── chart 1: val MSE ─────────────────────────────────────────────────────────
bar_group_chart(datasets, models, val_mse,
                ylabel="Validation MSE",
                title="Validation MSE by Model & Dataset",
                filename="val_mse_by_model.png",
                log=True)

# ── chart 2: boundary MSE ────────────────────────────────────────────────────
bar_group_chart(datasets, models, bnd_mse,
                ylabel="Boundary MSE  (|γ_max| < ε)",
                title="Boundary MSE by Model & Dataset",
                filename="boundary_mse_by_model.png",
                log=True)

# ── chart 3: topology distance ───────────────────────────────────────────────
bar_group_chart(datasets, models, topo_dist,
                ylabel="Topology Distance",
                title="Topology Distance by Model & Dataset",
                filename="topo_dist_by_model.png",
                fmt="{:.3f}")

# ── chart 4: speedup bar (log scale) ─────────────────────────────────────────
rt_rows = read_csv(RUNTIME_CSV)
# drop the solver row itself; keep neural models only
rt_rows = [r for r in rt_rows if r["Model"] != "Eigenvalue solver"]

rt_models  = [r["Model"]  for r in rt_rows]
rt_speedup = [float(r["Speedup"]) for r in rt_rows]

fig, ax = plt.subplots(figsize=(6, 4))
colors = [PALETTE.get(m, "#888888") for m in rt_models]
bars   = ax.bar(rt_models, rt_speedup, color=colors, width=0.6)
for bar, v in zip(bars, rt_speedup):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.05,
            f"{v:.0f}×", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_yscale("log")
ax.set_ylabel("Speedup vs Eigenvalue Solver  (×)", fontsize=10)
ax.set_title("Inference Speedup over Classical Solver", fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
path = os.path.join(OUT_DIR, "speedup_bar.png")
fig.savefig(path, dpi=120)
plt.close(fig)
print(f"  saved  {path}")

print("\nDone.  4 PNGs written to", OUT_DIR)
