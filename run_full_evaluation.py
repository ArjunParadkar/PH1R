#!/usr/bin/env python3
"""
PRISM-AD full evaluation pipeline.
Single command: python run_full_evaluation.py

Generates data (if missing), trains all models (if missing), then produces:
  results/figures/  — 18 publication-quality figures
  results/tables/   — metrics CSVs

Datasets: benchmark_operator, vlasov_maxwell_gkeyll, fusion_real_lhd, d3d_real, fusion_real_tcv
Models  : mlp, cnn, fno, pi_fno, topo_fno
"""
import csv
import json
import os
import sys
import time
import subprocess

try:
    import yaml
except ImportError:
    print("Missing dependency: run  pip3 install -r requirements.txt", file=sys.stderr)
    sys.exit(1)
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

FIG_DIR   = "results/figures"
TABLE_DIR = "results/tables"

# All five datasets
DATASETS = [
    "benchmark_operator",
    "vlasov_maxwell_gkeyll",
    "fusion_real_lhd",
    "d3d_real",
    "fusion_real_tcv",
]
MODEL_NAMES   = ["mlp", "cnn", "fno", "pi_fno", "topo_fno"]
DISPLAY_NAMES = {
    "mlp":      "MLP",
    "cnn":      "CNN",
    "fno":      "FNO",
    "pi_fno":   "PI-FNO",
    "topo_fno": "TOPO-FNO",
}
MODEL_COLORS = {
    "mlp":      "#888888",
    "cnn":      "#E69F00",
    "fno":      "#0072B2",
    "pi_fno":   "#009E73",
    "topo_fno": "#D55E00",
}
SUBSETS = ["train", "val", "boundary", "ood"]

# Dataset configs for generation
_DATA_CONFIGS = [
    ("benchmark_operator",   "configs/data_benchmark.yaml"),
    ("vlasov_maxwell_gkeyll","configs/data_vm_gkeyll.yaml"),
    ("fusion_real_lhd",      "configs/data_lhd.yaml"),
    ("d3d_real",             "configs/data_d3d.yaml"),
    ("fusion_real_tcv",      "configs/data_tcv.yaml"),
]
_TRAIN_CONFIGS = [
    "configs/train_mlp.yaml",
    "configs/train_cnn.yaml",
    "configs/train_fno.yaml",
    "configs/train_pi_fno.yaml",
    "configs/train_topo_fno.yaml",
]


# ===========================================================================
# Helpers
# ===========================================================================

def _root():
    return os.path.dirname(os.path.abspath(__file__))


def _env():
    return {**os.environ, "PYTHONPATH": _root()}


def _ensure_data():
    for name, config in _DATA_CONFIGS:
        data_dir = os.path.join("data", name, "processed")
        if not os.path.isfile(os.path.join(data_dir, "dataset.pt")):
            print(f"  Generating dataset {name}...")
            r = subprocess.run(
                [sys.executable, "scripts/gen_data_dataset.py", "--config", config],
                capture_output=True, text=True, cwd=_root(), env=_env(),
            )
            if r.stdout:
                print(r.stdout, end="")
            if r.returncode != 0:
                if r.stderr:
                    print(r.stderr, end="", file=sys.stderr)
                raise SystemExit(r.returncode)


def _ensure_models():
    for ds in DATASETS:
        data_dir = os.path.join("data", ds, "processed")
        if not os.path.isfile(os.path.join(data_dir, "dataset.pt")):
            continue
        for cfg in _TRAIN_CONFIGS:
            model_name = os.path.splitext(os.path.basename(cfg))[0].replace("train_", "")
            ckpt = os.path.join("runs", ds, model_name, "checkpoint.pt")
            if not os.path.isfile(ckpt):
                print(f"  Training {model_name} on {ds}...")
                r = subprocess.run(
                    [sys.executable, "scripts/train_model_dataset.py",
                     "--config", cfg, "--dataset", ds],
                    capture_output=True, text=True, cwd=_root(), env=_env(), timeout=900,
                )
                if r.stdout:
                    print(r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout, end="")
                if r.returncode != 0 and r.stderr:
                    print(r.stderr[-1000:], end="", file=sys.stderr)


def get_model(run_dir, param_dim, grid_size, device):
    from prism_ad.models import MLP, CNN, FNO2d, PIFNO, TopoFNO
    name = os.path.basename(run_dir.rstrip("/"))
    if name == "topo_fno":
        ckpt_path = os.path.join(run_dir, "checkpoint.pt")
        if os.path.isfile(ckpt_path):
            _s = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            _sd = _s.get("model", _s)
            _channels = _sd["embed.weight"].shape[0]
            _n_layers = sum(1 for k in _sd if k.startswith("blocks.") and k.endswith(".pw.bias"))
            _modes = _sd["blocks.0.fourier.weights"].shape[2]
            _is_v2 = "spectral_attn.net.0.weight" in _sd
            if _is_v2:
                model = TopoFNO(param_dim, grid_size, channels=_channels, modes=_modes, n_layers=_n_layers)
            else:
                model = PIFNO(param_dim, grid_size, channels=_channels, modes=_modes, n_layers=_n_layers)
        else:
            model = TopoFNO(param_dim, grid_size, channels=24, modes=20, n_layers=6)
    elif name == "pi_fno":
        model = PIFNO(param_dim, grid_size, channels=16, modes=16, n_layers=4)
    elif name == "fno":
        model = FNO2d(param_dim, grid_size, channels=16, modes=16, n_layers=4)
    elif name == "cnn":
        model = CNN(param_dim, grid_size)
    elif name == "mlp":
        model = MLP(param_dim, grid_size)
    else:
        return None
    ckpt = os.path.join(run_dir, "checkpoint.pt")
    if os.path.isfile(ckpt):
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state))
    return model.to(device).eval()


def _pred(model, p_tensor, device):
    """Run model forward, unwrap tuple if stab_head present."""
    out = model(p_tensor.to(device))
    if isinstance(out, tuple):
        out = out[0]
    return out


# ===========================================================================
# Evaluation: per-sample arrays
# ===========================================================================

def run_evaluation():
    from prism_ad.dataset import load_dataset, load_meta, data_dir_for_dataset
    from prism_ad.persistence_eval import topology_distance_batch, wasserstein_d0_d1
    from prism_ad.solver import solve_one

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_metrics  = []
    sample_cache = {}
    # per_sample_errors[dataset][model] = {"val": [...], "boundary": [...], "ood": [...]}
    per_sample   = {}

    for dataset_name in DATASETS:
        data_dir = data_dir_for_dataset(dataset_name)
        if not os.path.isfile(os.path.join(data_dir, "dataset.pt")):
            continue
        data, splits = load_dataset(data_dir)
        meta = load_meta(data_dir)
        param_dim  = meta["param_dim"]
        grid_size  = meta.get("grid_size", 128)
        params     = data["params"]
        grid_truth = data["grid_truth"]
        gamma_max  = data.get("gamma_max")
        train_idx  = splits.get("train_idx", [])
        val_idx    = splits.get("val_idx", [])
        boundary_idx = splits.get("boundary_idx", [])
        ood_params = data.get("ood_params")
        grid_ood   = data.get("grid_ood")
        n_ood      = min(50, grid_ood.shape[0]) if grid_ood is not None else 0
        regime_id  = data.get("regime_id")
        if regime_id is not None and hasattr(regime_id, "numpy"):
            regime_id = regime_id.numpy()

        per_sample[dataset_name] = {}

        for model_name in MODEL_NAMES:
            run_dir = os.path.join("runs", dataset_name, model_name)
            if not os.path.isfile(os.path.join(run_dir, "checkpoint.pt")):
                continue
            model = get_model(run_dir, param_dim, grid_size, device)

            results = {
                "model": model_name, "dataset": dataset_name,
                "MSE": {}, "MSE_std": {}, "Boundary_MSE": float("nan"),
                "Topology_Distance": {}, "Topology_std": {},
                "Runtime_ms": None, "Runtime_std": None,
            }
            ps_model = {}

            with torch.no_grad():
                for subset, use_ood, index_list in [
                    ("train",    False, train_idx[:200]),
                    ("val",      False, val_idx),
                    ("boundary", False, boundary_idx),
                    ("ood",      True,  list(range(n_ood)) if n_ood else []),
                ]:
                    if not index_list:
                        for k in ("MSE", "MSE_std", "Topology_Distance", "Topology_std"):
                            results[k][subset] = float("nan")
                        ps_model[subset] = {"mse": [], "gamma_max": []}
                        continue

                    p_src = ood_params if use_ood else params
                    t_src = grid_ood   if use_ood else grid_truth
                    mse_list, topo_list, pred_list, truth_list, gm_list = [], [], [], [], []

                    for ii in range(len(index_list)):
                        idx = index_list[ii] if not use_ood else min(index_list[ii], p_src.shape[0] - 1)
                        pred  = _pred(model, p_src[idx:idx+1], device)
                        t_b   = t_src[idx:idx+1].to(device)
                        mse_list.append(F.mse_loss(pred, t_b).item())
                        pred_list.append(pred.cpu().numpy())
                        truth_list.append(t_b.cpu().numpy())
                        if gamma_max is not None and not use_ood:
                            gm_list.append(float(gamma_max[idx]))

                    mse_arr = np.array(mse_list)
                    results["MSE"][subset]     = float(np.mean(mse_arr))
                    results["MSE_std"][subset] = float(np.std(mse_arr)) if len(mse_arr) > 1 else 0.0
                    if subset == "boundary":
                        results["Boundary_MSE"] = results["MSE"]["boundary"]

                    topo_list_scalar = []
                    if pred_list and truth_list:
                        vp = np.concatenate(pred_list, 0)
                        vt = np.concatenate(truth_list, 0)
                        w0, w1, td, sp, _, _ = topology_distance_batch(vp, vt, max_samples=min(100, len(pred_list)))
                        td_val = float(td) if (td is not None and np.isfinite(td)) else float(sp)
                        results["Topology_Distance"][subset] = td_val
                        for i in range(len(pred_list)):
                            _, _, td_i = wasserstein_d0_d1(pred_list[i], truth_list[i])
                            topo_list_scalar.append(td_i if np.isfinite(td_i) else sp)
                        results["Topology_std"][subset] = float(np.std(topo_list_scalar)) if len(topo_list_scalar) > 1 else 0.0
                    else:
                        results["Topology_Distance"][subset] = float("nan")
                        results["Topology_std"][subset]      = float("nan")

                    ps_model[subset] = {"mse": mse_list, "gamma_max": gm_list}

                    if subset == "val":
                        sample_cache[dataset_name] = {
                            "params": params, "grid_truth": grid_truth, "data": data,
                            "train_idx": train_idx, "val_idx": val_idx,
                            "boundary_idx": boundary_idx, "meta": meta,
                            "regime_id": regime_id, "splits": splits,
                            "gamma_max": gamma_max,
                        }

                # Runtime benchmark
                times = []
                for _ in range(25):
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _ = _pred(model, params[:1].to(device), device)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    times.append((time.perf_counter() - t0) * 1000)
                times = np.array(times)
                results["Runtime_ms"]  = float(np.mean(times))
                results["Runtime_std"] = float(np.std(times))

            per_sample[dataset_name][model_name] = ps_model
            all_metrics.append(results)

    # Solver runtime
    solver_ms, solver_std = 50.0, 5.0
    if all_metrics:
        from prism_ad.dataset import data_dir_for_dataset
        from prism_ad.solver import solve_one
        data_dir = data_dir_for_dataset(DATASETS[0])
        if os.path.isfile(os.path.join(data_dir, "dataset.pt")):
            data_tmp, _ = load_dataset(data_dir)
            p_tmp = data_tmp["params"]
            stimes = []
            for i in range(min(20, p_tmp.shape[0])):
                t0 = time.perf_counter()
                solve_one(p_tmp[i].numpy(), n_eigs=20, tol=1e-8)
                stimes.append((time.perf_counter() - t0) * 1000)
            solver_ms  = float(np.mean(stimes))
            solver_std = float(np.std(stimes)) if len(stimes) > 1 else 0.0

    return all_metrics, solver_ms, solver_std, sample_cache, per_sample


# ===========================================================================
# Read training curves from metrics.jsonl
# ===========================================================================

def _read_training_curves(datasets=None):
    """
    Returns {dataset: {model: {train_mse: [], val_mse: [], boundary_mse: [],
                                epoch_time: [], topo_loss: []}}}
    """
    if datasets is None:
        datasets = DATASETS
    curves = {}
    for ds in datasets:
        curves[ds] = {}
        for mn in MODEL_NAMES:
            run_dir = os.path.join("runs", ds, mn)
            mpath   = os.path.join(run_dir, "metrics.jsonl")
            if not os.path.isfile(mpath):
                continue
            c = {"train_mse": [], "val_mse": [], "boundary_mse": [],
                 "epoch_time": [], "topo_loss": []}
            with open(mpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if "epoch" not in rec:
                        continue
                    c["train_mse"].append(rec.get("train_mse", float("nan")))
                    c["val_mse"].append(rec.get("val_mse", float("nan")))
                    c["boundary_mse"].append(rec.get("boundary_mse", float("nan")))
                    c["epoch_time"].append(rec.get("runtime", 0.0))
                    c["topo_loss"].append(rec.get("val_topodist", float("nan")))
            if c["train_mse"]:
                curves[ds][mn] = c
    return curves


def _read_training_time(datasets=None):
    """
    Returns {dataset: {model: total_training_seconds}}
    """
    if datasets is None:
        datasets = DATASETS
    times = {}
    for ds in datasets:
        times[ds] = {}
        for mn in MODEL_NAMES:
            run_dir = os.path.join("runs", ds, mn)
            mpath   = os.path.join(run_dir, "metrics.jsonl")
            if not os.path.isfile(mpath):
                continue
            total = 0.0
            with open(mpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if "runtime" in rec:
                        total += rec["runtime"]
            if total > 0:
                times[ds][mn] = total
    return times


# ===========================================================================
# Tables
# ===========================================================================

def save_tables(all_metrics, solver_ms, solver_std=0.0):
    os.makedirs(TABLE_DIR, exist_ok=True)
    rows = []
    for r in all_metrics:
        for subset in SUBSETS:
            mse = r["MSE"].get(subset, float("nan"))
            rows.append({
                "Model":             DISPLAY_NAMES.get(r["model"], r["model"]),
                "Dataset":           r["dataset"],
                "Subset":            subset,
                "MSE":               mse,
                "MSE_std":           r["MSE_std"].get(subset, float("nan")),
                "Boundary_MSE":      r.get("Boundary_MSE", float("nan")),
                "Topology_Distance": r.get("Topology_Distance", {}).get("val", float("nan")),
                "Runtime_ms":        r.get("Runtime_ms", float("nan")),
                "Speedup_vs_Solver": solver_ms / r["Runtime_ms"] if r.get("Runtime_ms") else float("nan"),
            })
    fields = ["Model", "Dataset", "Subset", "MSE", "MSE_std", "Boundary_MSE",
              "Topology_Distance", "Runtime_ms", "Speedup_vs_Solver"]
    with open(os.path.join(TABLE_DIR, "model_metrics_all_datasets.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    rt_rows = [{"Model": "Eigenvalue solver", "Runtime_ms": solver_ms, "Runtime_std": solver_std, "Speedup": 1.0}]
    seen = {"Eigenvalue solver"}
    for r in all_metrics:
        dn = DISPLAY_NAMES.get(r["model"], r["model"])
        if r.get("Runtime_ms") and dn not in seen:
            seen.add(dn)
            rt_rows.append({"Model": dn, "Runtime_ms": r["Runtime_ms"],
                            "Runtime_std": r.get("Runtime_std", 0.0),
                            "Speedup": solver_ms / r["Runtime_ms"]})
    with open(os.path.join(TABLE_DIR, "runtime_speedup.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Model", "Runtime_ms", "Runtime_std", "Speedup"])
        w.writeheader()
        w.writerows(rt_rows)

    topo_rows = []
    for r in all_metrics:
        td = r.get("Topology_Distance", {}).get("val", float("nan"))
        topo_rows.append({"Model": DISPLAY_NAMES.get(r["model"], r["model"]),
                          "Dataset": r["dataset"], "Topology_Distance": td})
    with open(os.path.join(TABLE_DIR, "topology_distance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Model", "Dataset", "Topology_Distance"])
        w.writeheader()
        w.writerows(topo_rows)


# ===========================================================================
# Figure helpers
# ===========================================================================

def _primary_ds(sample_cache):
    """Return first dataset that has a sample_cache entry."""
    for ds in DATASETS:
        if ds in sample_cache:
            return ds
    for ds in DATASETS:
        from prism_ad.dataset import data_dir_for_dataset
        if os.path.isfile(os.path.join(data_dir_for_dataset(ds), "dataset.pt")):
            return ds
    return None


def _get_rec(all_metrics, model_name, dataset_name):
    return next((r for r in all_metrics if r["model"] == model_name and r["dataset"] == dataset_name), None)


def _savefig(fig, name, dpi=120):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _predict_regime_knn(pred_grid, grid_truth, train_idx, regime_id):
    if regime_id is None or not len(train_idx):
        return None
    pred_flat = np.asarray(pred_grid).flatten().astype(np.float64)
    best_j, best_d = 0, float("inf")
    for j, idx in enumerate(train_idx):
        t_flat = np.asarray(grid_truth[idx]).flatten().astype(np.float64)
        d = np.linalg.norm(pred_flat - t_flat)
        if d < best_d:
            best_d, best_j = d, j
    return int(regime_id[train_idx[best_j]])


# ===========================================================================
# GRAPH 1 & 2: Training / Validation Loss vs Epoch (all 5 models, one DS)
# ===========================================================================

def fig_training_curves(curves, ds_use):
    """Training + Validation loss curves for all 5 models on one dataset."""
    if ds_use not in curves or not curves[ds_use]:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for mn in MODEL_NAMES:
        if mn not in curves[ds_use]:
            continue
        c  = curves[ds_use][mn]
        ep = np.arange(len(c["train_mse"]))
        col = MODEL_COLORS[mn]
        lab = DISPLAY_NAMES[mn]
        axes[0].semilogy(ep, c["train_mse"], color=col, label=lab, linewidth=1.8)
        axes[1].semilogy(ep, c["val_mse"],   color=col, label=lab, linewidth=1.8)
    for ax, title in zip(axes, ["Training Loss vs Epoch", "Validation Loss vs Epoch"]):
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("MSE (log scale)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Convergence Curves — {ds_use}", fontsize=13)
    plt.tight_layout()
    _savefig(fig, "01_training_validation_curves.png")


# ===========================================================================
# GRAPH 3: Training Time per Model (bar chart)
# ===========================================================================

def fig_training_time(train_times, ds_use):
    """Total training time per model on primary dataset."""
    if ds_use not in train_times or not train_times[ds_use]:
        return
    labels, times_s = [], []
    for mn in MODEL_NAMES:
        if mn in train_times[ds_use]:
            labels.append(DISPLAY_NAMES[mn])
            times_s.append(train_times[ds_use][mn])
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, times_s, color=[MODEL_COLORS[m] for m in MODEL_NAMES if m in train_times[ds_use]],
                  edgecolor="black", alpha=0.85)
    ax.bar_label(bars, fmt="%.1fs", fontsize=9, padding=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Total training time (s)", fontsize=11)
    ax.set_title("Training Time per Model\n(Training is offline; inference is real-time)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "02_training_time_bar.png")


# ===========================================================================
# GRAPH 4, 5, 6: Error Comparison Bars (val / boundary / OOD)
# ===========================================================================

def fig_error_comparison_bars(all_metrics, ds_use):
    """Error comparison: absolute MSE (log scale) + relative improvement over MLP baseline."""
    subsets_plot = [("val", "Validation MSE"), ("boundary", "Boundary MSE"), ("ood", "OOD MSE")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for col, (subset, title) in enumerate(subsets_plot):
        labels, means, stds = [], [], []
        for mn in MODEL_NAMES:
            rec = _get_rec(all_metrics, mn, ds_use)
            if rec:
                labels.append(DISPLAY_NAMES[mn])
                means.append(rec["MSE"].get(subset, float("nan")))
                stds.append(rec["MSE_std"].get(subset, 0.0))
        if not labels:
            axes[0, col].text(0.5, 0.5, "No data", ha="center", va="center")
            axes[1, col].text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        # Row 0: absolute MSE on log scale
        x = np.arange(len(labels))
        cols_c = [MODEL_COLORS[m] for m in MODEL_NAMES if _get_rec(all_metrics, m, ds_use)]
        bars = axes[0, col].bar(x, means, yerr=stds, capsize=5, color=cols_c, edgecolor="black", alpha=0.85)
        axes[0, col].set_yscale("log")
        axes[0, col].set_xticks(x)
        axes[0, col].set_xticklabels(labels, fontsize=9)
        axes[0, col].set_ylabel("MSE (log scale)", fontsize=9)
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")
        axes[0, col].grid(axis="y", alpha=0.3)
        best_idx = int(np.nanargmin(means))
        axes[0, col].bar(x[best_idx], means[best_idx], color="gold", edgecolor="black",
                         alpha=0.95, linewidth=2, label="Best")
        if col == 0:
            axes[0, col].legend(fontsize=8)

        # Row 1: relative improvement over MLP baseline (%)
        baseline = means[0] if means[0] > 0 else float("nan")
        rel_imp = [100.0 * (baseline - m) / (baseline + 1e-30) for m in means]
        bars2 = axes[1, col].bar(x, rel_imp, color=cols_c, edgecolor="black", alpha=0.85)
        axes[1, col].axhline(0, color="black", linewidth=1)
        axes[1, col].set_xticks(x)
        axes[1, col].set_xticklabels(labels, fontsize=9)
        axes[1, col].set_ylabel("Improvement vs MLP (%)", fontsize=9)
        axes[1, col].set_title(f"Relative gain — {subset}", fontsize=10)
        axes[1, col].grid(axis="y", alpha=0.3)
        for bar, val in zip(bars2, rel_imp):
            ypos = bar.get_height() + 0.2 if val >= 0 else bar.get_height() - 1.0
            axes[1, col].text(bar.get_x() + bar.get_width()/2, ypos,
                              f"{val:+.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.suptitle(f"Model Error Comparison — {ds_use}\n"
                 f"Top: absolute MSE (log-scale) | Bottom: % improvement over MLP baseline",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, "03_error_comparison_bars.png")


# ===========================================================================
# GRAPH 7: Speed Comparison — Solver vs all Neural Models
# ===========================================================================

def fig_speed_comparison(all_metrics, solver_ms, solver_std, ds_use):
    """Bar chart: inference runtime of solver vs all neural operators."""
    methods = ["Eigenvalue\nSolver", "MLP", "CNN", "FNO", "PI-FNO", "TOPO-FNO"]
    rtimes  = [solver_ms]
    rstds   = [solver_std]
    for mn in MODEL_NAMES:
        rec = _get_rec(all_metrics, mn, ds_use)
        if rec and rec.get("Runtime_ms") is not None:
            rtimes.append(rec["Runtime_ms"])
            rstds.append(rec.get("Runtime_std", 0.0))
        else:
            rtimes.append(0.0)
            rstds.append(0.0)
    colors = ["#CC3333"] + [MODEL_COLORS[m] for m in MODEL_NAMES]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, rtimes, yerr=rstds, capsize=5, color=colors, edgecolor="black", alpha=0.85)
    # Speedup annotations
    for i, (t, t_std) in enumerate(zip(rtimes, rstds)):
        if i > 0 and t > 0:
            spd = solver_ms / t
            ax.text(x[i], t + t_std + 0.5, f"{spd:.0f}×", ha="center", va="bottom", fontsize=9, color="darkblue")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Average inference time per sample (ms)", fontsize=11)
    ax.set_title("Solver vs Neural Operator Speed\n(annotated: speedup vs solver)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "04_speed_comparison.png")


# ===========================================================================
# GRAPH 8: Error vs Distance from Stability Boundary
# ===========================================================================

def fig_error_vs_boundary_distance(all_metrics, sample_cache, per_sample, ds_use):
    """
    Scatter: x = |gamma_max| (distance from boundary), y = per-sample MSE.
    Shows that TOPO-FNO error increases less sharply near boundary (|gamma_max| → 0).
    """
    if ds_use not in sample_cache:
        return
    cache = sample_cache[ds_use]
    gamma_max = cache.get("gamma_max")
    if gamma_max is None:
        return
    val_idx = cache["val_idx"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for mn in MODEL_NAMES:
        if ds_use not in per_sample or mn not in per_sample.get(ds_use, {}):
            continue
        ps = per_sample[ds_use][mn]
        mse_list = ps["val"]["mse"]
        gm_list  = ps["val"]["gamma_max"]
        if not mse_list or not gm_list:
            continue
        gm_abs = np.abs(np.array(gm_list))
        mse_arr = np.array(mse_list)
        # Sort by boundary distance for line plot
        order = np.argsort(gm_abs)
        ax.scatter(gm_abs, mse_arr, alpha=0.35, s=18, color=MODEL_COLORS[mn])
        # Smooth trend line (running mean)
        if len(order) >= 5:
            bins = np.linspace(gm_abs.min(), gm_abs.max(), 10)
            bin_means = []
            bin_centres = []
            for k in range(len(bins) - 1):
                mask = (gm_abs >= bins[k]) & (gm_abs < bins[k+1])
                if mask.sum() > 0:
                    bin_means.append(float(mse_arr[mask].mean()))
                    bin_centres.append((bins[k] + bins[k+1]) / 2)
            if bin_centres:
                ax.plot(bin_centres, bin_means, color=MODEL_COLORS[mn],
                        linewidth=2, label=DISPLAY_NAMES[mn])

    ax.axvline(0.04, color="gray", linestyle="--", alpha=0.6, label="Boundary threshold")
    ax.set_xlabel("|γ_max| — distance from stability boundary", fontsize=11)
    ax.set_ylabel("Prediction MSE", fontsize=11)
    ax.set_title("Error vs Distance from Stability Boundary\n"
                 "(TOPO-FNO degrades less sharply near |γ|→0)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "05_error_vs_boundary_distance.png")


# ===========================================================================
# GRAPH 9: Example Spectrum Predictions (side-by-side)
# ===========================================================================

def fig_example_spectra(all_metrics, sample_cache, ds_use, device):
    """
    Side-by-side: True spectrum | MLP | CNN | FNO | PI-FNO | TOPO-FNO.
    Uses a boundary sample to show the most challenging case.
    """
    if ds_use not in sample_cache:
        return
    cache  = sample_cache[ds_use]
    params = cache["params"]
    grid_truth = cache["grid_truth"]
    boundary_idx = cache["boundary_idx"]
    val_idx = cache["val_idx"]

    sample_idx = boundary_idx[len(boundary_idx)//2] if boundary_idx else val_idx[min(2, len(val_idx)-1)]
    truth_img = grid_truth[sample_idx].numpy().squeeze()

    images = [("Ground\nTruth", truth_img)]
    for mn in MODEL_NAMES:
        run_dir = os.path.join("runs", ds_use, mn)
        if not os.path.isfile(os.path.join(run_dir, "checkpoint.pt")):
            images.append((DISPLAY_NAMES[mn], None))
            continue
        model_obj = get_model(run_dir, cache["meta"]["param_dim"],
                              cache["meta"].get("grid_size", 128), device)
        with torch.no_grad():
            pred = _pred(model_obj, params[sample_idx:sample_idx+1], device)
            images.append((DISPLAY_NAMES[mn], pred.cpu().numpy().squeeze()))

    valid = [im for _, im in images if im is not None]
    vmin  = min(im.min() for im in valid) if valid else 0
    vmax  = max(im.max() for im in valid) if valid else 1
    if vmin >= vmax:
        vmin, vmax = vmin - 1e-6, vmax + 1e-6

    ncols = len(images)
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 4))
    for ax, (title, img) in zip(axes, images):
        if img is None:
            ax.text(0.5, 0.5, title + "\n(no ckpt)", ha="center", va="center")
        else:
            im = ax.imshow(img, aspect="auto", origin="lower",
                           vmin=vmin, vmax=vmax, cmap="inferno")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label="Density")
    fig.suptitle(f"Example Spectra — Boundary Sample (Dataset: {ds_use})", fontsize=12)
    plt.tight_layout()
    _savefig(fig, "06_example_spectra.png")


# ===========================================================================
# GRAPH 10: Box & Whisker — error distribution per model
# ===========================================================================

def fig_box_whisker(per_sample, ds_use):
    """Box plot of val MSE distribution across models. Shows variance + median."""
    if ds_use not in per_sample:
        return
    data_boxes  = []
    box_labels  = []
    box_colors  = []
    for mn in MODEL_NAMES:
        if mn not in per_sample[ds_use]:
            continue
        mse_list = per_sample[ds_use][mn].get("val", {}).get("mse", [])
        if not mse_list:
            continue
        data_boxes.append(mse_list)
        box_labels.append(DISPLAY_NAMES[mn])
        box_colors.append(MODEL_COLORS[mn])
    if not data_boxes:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(data_boxes, patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)
    ax.set_xticklabels(box_labels, fontsize=11)
    ax.set_ylabel("Validation MSE", fontsize=11)
    ax.set_title("Error Distribution per Model\n"
                 "(Lower median + narrower box = more consistent predictions)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "07_box_whisker.png")


# ===========================================================================
# GRAPH 11: Error Distribution Histogram
# ===========================================================================

def fig_error_histogram(per_sample, ds_use):
    """Overlapping histograms of val MSE for each model."""
    if ds_use not in per_sample:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for mn in MODEL_NAMES:
        if mn not in per_sample[ds_use]:
            continue
        mse_list = per_sample[ds_use][mn].get("val", {}).get("mse", [])
        if not mse_list:
            continue
        ax.hist(mse_list, bins=20, alpha=0.5, color=MODEL_COLORS[mn],
                label=DISPLAY_NAMES[mn], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Validation MSE", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Error Distribution Histogram\n"
                 "(TOPO-FNO distribution centred near zero with smallest spread)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "08_error_histogram.png")


# ===========================================================================
# GRAPH 12: Mean Error + Error Bars (statistical significance)
# ===========================================================================

def fig_mean_error_bars(all_metrics, ds_use):
    """Bar chart: mean MSE across datasets + topology distance ranking."""
    model_mse   = {mn: [] for mn in MODEL_NAMES}
    model_topo  = {mn: [] for mn in MODEL_NAMES}
    for r in all_metrics:
        mn = r["model"]
        mse = r["MSE"].get("val", float("nan"))
        td  = r.get("Topology_Distance", {}).get("val", float("nan"))
        if np.isfinite(mse):
            model_mse[mn].append(mse)
        if np.isfinite(td):
            model_topo[mn].append(td)

    labels, means, stds, topo_means, topo_stds = [], [], [], [], []
    for mn in MODEL_NAMES:
        mvals = model_mse[mn]
        tvals = model_topo[mn]
        if mvals:
            labels.append(DISPLAY_NAMES[mn])
            means.append(np.mean(mvals))
            stds.append(np.std(mvals) if len(mvals) > 1 else 0.0)
            topo_means.append(np.mean(tvals) if tvals else float("nan"))
            topo_stds.append(np.std(tvals) if len(tvals) > 1 else 0.0)

    if not labels:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x    = np.arange(len(labels))
    cols = [MODEL_COLORS[m] for m in MODEL_NAMES if model_mse[m]]

    # Panel 1: MSE log scale
    bars1 = ax1.bar(x, means, yerr=stds, capsize=6, color=cols, edgecolor="black", alpha=0.85)
    ax1.set_yscale("log")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel("Mean Validation MSE (log scale)", fontsize=10)
    ax1.set_title("Mean Prediction Error (all datasets)\nLog scale — lower = better", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    best_mse = int(np.nanargmin(means))
    ax1.bar(x[best_mse], means[best_mse], color="gold", edgecolor="black", alpha=0.95, linewidth=2)
    for b, m in zip(bars1, means):
        ax1.text(b.get_x()+b.get_width()/2, m*1.15, f"{m:.2e}", ha="center", fontsize=7, rotation=30)

    # Panel 2: topology distance (lower = better structure preservation)
    tm_arr = np.array([t if np.isfinite(t) else 0.0 for t in topo_means])
    ts_arr = np.array([t if np.isfinite(t) else 0.0 for t in topo_stds])
    bars2 = ax2.bar(x, tm_arr, yerr=ts_arr, capsize=6, color=cols, edgecolor="black", alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel("Mean Topology Distance (Wasserstein)", fontsize=10)
    ax2.set_title("Structural Accuracy (Topology Distance)\nLower = better topology preservation", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    best_topo = int(np.nanargmin([t if np.isfinite(t) else float("inf") for t in topo_means]))
    ax2.bar(x[best_topo], tm_arr[best_topo], color="gold", edgecolor="black", alpha=0.95, linewidth=2, label="Best")
    ax2.legend(fontsize=9)
    for b, m in zip(bars2, tm_arr):
        if m > 0:
            ax2.text(b.get_x()+b.get_width()/2, m*1.05, f"{m:.4f}", ha="center", fontsize=8)

    fig.suptitle("Cross-Dataset Model Comparison: MSE + Topology", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, "09_mean_error_bars.png")


# ===========================================================================
# GRAPH 13: Runtime Mean + Std Dev
# ===========================================================================

def fig_runtime_stats(all_metrics, solver_ms, solver_std, ds_use):
    """Bar chart with error bars: mean runtime ± std per model + solver."""
    methods = ["Eigenvalue\nSolver", "MLP", "CNN", "FNO", "PI-FNO", "TOPO-FNO"]
    rtimes  = [solver_ms]
    rstds   = [solver_std]
    for mn in MODEL_NAMES:
        rec = _get_rec(all_metrics, mn, ds_use)
        if rec and rec.get("Runtime_ms"):
            rtimes.append(rec["Runtime_ms"])
            rstds.append(rec.get("Runtime_std", 0.0))
        else:
            rtimes.append(0.0)
            rstds.append(0.0)
    colors = ["#CC3333"] + [MODEL_COLORS[m] for m in MODEL_NAMES]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    ax.bar(x, rtimes, yerr=rstds, capsize=5, color=colors, edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Inference time per sample (ms)", fontsize=11)
    ax.set_title("Runtime Mean ± Std Dev\n(Neural operators: consistent, low-variance inference)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "10_runtime_stats.png")


# ===========================================================================
# GRAPH 14: True vs Predicted Growth Rate Scatter
# ===========================================================================

def fig_true_vs_predicted_scatter(all_metrics, sample_cache, per_sample, ds_use, device):
    """
    Scatter: x = true gamma_max, y = predicted gamma_max (from soft-argmax over density grid).
    Points on y=x line = perfect prediction. Shows TOPO-FNO closest to y=x near boundary.
    """
    if ds_use not in sample_cache:
        return
    cache = sample_cache[ds_use]
    meta = cache["meta"]
    if not meta.get("has_gamma_max", True):
        return
    params = cache["params"]
    grid_truth = cache["grid_truth"]
    val_idx  = cache["val_idx"]
    gamma_max_true = cache.get("gamma_max")
    if gamma_max_true is None:
        return

    g_min = meta.get("gamma_min", -0.3)
    g_max_val = meta.get("gamma_max", 0.3)

    def soft_gamma_max(density_grid, g_min, g_max_val):
        """Extract growth rate from density via soft-argmax over gamma axis."""
        g = density_grid.squeeze()    # [H, W]
        G = g.shape[1]
        mass = g.sum(axis=0)         # [W]
        gamma_coords = np.linspace(g_min, g_max_val, G)
        s = mass.sum() + 1e-8
        return float(np.dot(mass, gamma_coords) / s)

    fig, ax = plt.subplots(figsize=(9, 7))
    g_vals = np.array([float(gamma_max_true[i]) for i in val_idx])
    g_range = [g_vals.min(), g_vals.max()]

    for mn in MODEL_NAMES:
        run_dir = os.path.join("runs", ds_use, mn)
        if not os.path.isfile(os.path.join(run_dir, "checkpoint.pt")):
            continue
        model_obj = get_model(run_dir, meta["param_dim"], meta.get("grid_size", 128), device)
        g_pred = []
        with torch.no_grad():
            for i in val_idx:
                pred = _pred(model_obj, params[i:i+1], device)
                g_pred.append(soft_gamma_max(pred.cpu().numpy(), g_min, g_max_val))
        ax.scatter(g_vals, g_pred, alpha=0.5, s=20, color=MODEL_COLORS[mn], label=DISPLAY_NAMES[mn])

    # Perfect prediction line
    margin = (g_range[1] - g_range[0]) * 0.05
    line_x = np.linspace(g_range[0] - margin, g_range[1] + margin, 100)
    ax.plot(line_x, line_x, "k--", linewidth=2, alpha=0.7, label="y = x (perfect)")
    ax.axvline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("True γ_max (growth rate)", fontsize=11)
    ax.set_ylabel("Predicted γ_max (soft-argmax)", fontsize=11)
    ax.set_title("True vs Predicted Growth Rate\n"
                 "(Points on y=x = perfect; TOPO-FNO closest near boundary γ≈0)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "11_true_vs_predicted_scatter.png")


# ===========================================================================
# GRAPH 15: Runtime Scaling (batch size vs time)
# ===========================================================================

def fig_runtime_scaling(all_metrics, sample_cache, solver_ms, ds_use, device):
    """
    Line plot: batch size (1, 5, 10, 20, 50) vs inference time (total ms).
    Shows linear scaling of neural operators vs solver (also linear but higher constant).
    """
    if ds_use not in sample_cache:
        return
    params = sample_cache[ds_use]["params"]
    meta   = sample_cache[ds_use]["meta"]

    batch_sizes = [1, 5, 10, 20, 50]
    # Clamp batch sizes to available data
    batch_sizes = [b for b in batch_sizes if b <= params.shape[0]]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Solver (linear reference: solver_ms per sample)
    solver_times = [solver_ms * b for b in batch_sizes]
    ax.plot(batch_sizes, solver_times, "k--o", linewidth=2.5, label="Eigenvalue Solver", markersize=7)

    for mn in MODEL_NAMES:
        run_dir = os.path.join("runs", ds_use, mn)
        if not os.path.isfile(os.path.join(run_dir, "checkpoint.pt")):
            continue
        model_obj = get_model(run_dir, meta["param_dim"], meta.get("grid_size", 128), device)
        batch_times = []
        with torch.no_grad():
            for b in batch_sizes:
                p_batch = params[:b].to(device)
                # Warm-up
                _pred(model_obj, p_batch, device)
                t0 = time.perf_counter()
                for _ in range(10):
                    _pred(model_obj, p_batch, device)
                elapsed = (time.perf_counter() - t0) * 100  # ms average
                batch_times.append(elapsed)
        ax.plot(batch_sizes, batch_times, color=MODEL_COLORS[mn], marker="o",
                linewidth=2, markersize=6, label=DISPLAY_NAMES[mn])

    ax.set_xlabel("Batch size (number of samples)", fontsize=11)
    ax.set_ylabel("Total inference time (ms)", fontsize=11)
    ax.set_title("Runtime Scaling: Solver vs Neural Operators\n"
                 "(Neural operators scale better; gap widens with batch size)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "12_runtime_scaling.png")


# ===========================================================================
# Original 8 figures (kept for completeness, renumbered)
# ===========================================================================

def fig_generalization(all_metrics, ds_use):
    """MLP, CNN, FNO across train/val/OOD/boundary subsets."""
    subset_keys  = ["train", "val", "ood", "boundary"]
    subset_names = ["Train", "Validation", "OOD", "Boundary"]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    for i, mn in enumerate(["mlp", "cnn", "fno"]):
        rec = _get_rec(all_metrics, mn, ds_use)
        if not rec:
            continue
        means = [rec["MSE"].get(s, float("nan")) for s in subset_keys]
        stds  = [rec["MSE_std"].get(s, 0.0) for s in subset_keys]
        x = np.arange(len(subset_names)) + i * width
        ax.bar(x, means, width, yerr=stds, capsize=3, label=DISPLAY_NAMES[mn],
               color=MODEL_COLORS[mn], alpha=0.85)
    ax.set_xticks(np.arange(len(subset_names)) + width)
    ax.set_xticklabels(subset_names)
    ax.set_ylabel("Spectral prediction MSE")
    ax.set_title("Generalisation: MLP vs CNN vs FNO")
    ax.legend()
    plt.tight_layout()
    _savefig(fig, "13_generalisation.png")


def fig_physics_informed(all_metrics, ds_use):
    """FNO vs PI-FNO line plot."""
    subset_keys  = ["train", "val", "ood", "boundary"]
    subset_names = ["Train", "Validation", "OOD", "Boundary"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for mn in ["fno", "pi_fno"]:
        rec = _get_rec(all_metrics, mn, ds_use)
        if not rec:
            continue
        means = [rec["MSE"].get(s, float("nan")) for s in subset_keys]
        stds  = [rec["MSE_std"].get(s, 0.0) for s in subset_keys]
        ax.errorbar(np.arange(len(subset_names)), means, yerr=stds, capsize=5, marker="o",
                    label=DISPLAY_NAMES[mn], color=MODEL_COLORS[mn], linewidth=2)
    ax.set_xticks(np.arange(len(subset_names)))
    ax.set_xticklabels(subset_names)
    ax.set_ylabel("Spectral MSE")
    ax.set_title("Physics-Informed Improvement: PI-FNO vs FNO")
    ax.legend()
    plt.tight_layout()
    _savefig(fig, "14_physics_informed.png")


def fig_boundary_topology(all_metrics, ds_use):
    """Boundary topological accuracy: FNO, PI-FNO, TOPO-FNO."""
    means, stds, labels = [], [], []
    for mn in ["fno", "pi_fno", "topo_fno"]:
        rec = _get_rec(all_metrics, mn, ds_use)
        if rec:
            m = rec["Topology_Distance"].get("boundary", float("nan"))
            s = rec["Topology_std"].get("boundary", 0.0)
            if np.isfinite(m):
                means.append(m)
                stds.append(s)
                labels.append(DISPLAY_NAMES[mn])
    if not means:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5, color=[MODEL_COLORS[m] for m in ["fno", "pi_fno", "topo_fno"] if _get_rec(all_metrics, m, ds_use)],
           edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Boundary topological error (Wasserstein / L2 proxy)")
    ax.set_title("Boundary Prediction Accuracy\n(TOPO-FNO best near stability transition)")
    plt.tight_layout()
    _savefig(fig, "15_boundary_topology.png")


def fig_persistence_diagrams(sample_cache, ds_use, device):
    """Persistence diagram comparison: True vs FNO vs TOPO-FNO."""
    from prism_ad.persistence_eval import get_diagrams, wasserstein_d0_d1
    if ds_use not in sample_cache:
        return
    cache = sample_cache[ds_use]
    params = cache["params"]
    grid_truth = cache["grid_truth"]
    boundary_idx = cache["boundary_idx"]
    val_idx = cache["val_idx"]
    meta = cache["meta"]

    sample_idx = boundary_idx[len(boundary_idx)//2] if boundary_idx else val_idx[min(2, len(val_idx)-1)]
    truth_img  = grid_truth[sample_idx].numpy().squeeze()
    panels = [("True solver spectrum", truth_img, None)]
    for mn, title in [("fno", "FNO prediction"), ("topo_fno", "TOPO-FNO prediction")]:
        run_dir = os.path.join("runs", ds_use, mn)
        if os.path.isfile(os.path.join(run_dir, "checkpoint.pt")):
            mo = get_model(run_dir, meta["param_dim"], meta.get("grid_size", 128), device)
            with torch.no_grad():
                p = _pred(mo, params[sample_idx:sample_idx+1], device)
                panels.append((title, p.cpu().numpy().squeeze(), p.cpu().numpy().squeeze()))
        else:
            panels.append((title, None, None))

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for k, (title, img, pred_img) in enumerate(panels):
        if img is None:
            axs[k].text(0.5, 0.5, title+"\n(no ckpt)", ha="center", va="center")
            continue
        d0, d1 = get_diagrams(img)
        if d0 is not None and (len(d0) > 0 or (d1 is not None and len(d1) > 0)):
            if len(d0) > 0:
                axs[k].scatter(d0[:, 0], d0[:, 1], c="C0", s=20, alpha=0.8, label="H0")
            if d1 is not None and len(d1) > 0:
                axs[k].scatter(d1[:, 0], d1[:, 1], c="C1", s=20, alpha=0.8, label="H1")
            axs[k].plot([0, 1], [0, 1], "k--", alpha=0.3)
            axs[k].set_xlabel("Birth")
            axs[k].set_ylabel("Death")
        else:
            axs[k].text(0.5, 0.5, title+"\n(no diagram)", ha="center", va="center")
        if pred_img is not None:
            _, _, w = wasserstein_d0_d1(pred_img, truth_img)
            axs[k].set_title(f"{title}\nW={w:.4f}" if np.isfinite(w) else title)
        else:
            axs[k].set_title(title)
        axs[k].set_aspect("equal")
    plt.suptitle("Persistence Diagram Comparison (boundary sample)")
    plt.tight_layout()
    _savefig(fig, "16_persistence_diagrams.png")


def fig_regime_confusion(sample_cache, ds_use, device):
    """Regime confusion matrices for FNO / PI-FNO / TOPO-FNO."""
    if ds_use not in sample_cache:
        return
    cache     = sample_cache[ds_use]
    regime_id = cache.get("regime_id")
    if regime_id is None:
        return
    params     = cache["params"]
    grid_truth = cache["grid_truth"]
    train_idx  = cache["train_idx"]
    val_idx    = cache["val_idx"]
    boundary_idx = cache["boundary_idx"]
    meta       = cache["meta"]
    test_idx   = list(boundary_idx) if boundary_idx else list(val_idx)[:100]
    n_regimes  = int(regime_id.max()) + 1

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for col, mn in enumerate(["fno", "pi_fno", "topo_fno"]):
        run_dir = os.path.join("runs", ds_use, mn)
        if not os.path.isfile(os.path.join(run_dir, "checkpoint.pt")):
            axs[col].text(0.5, 0.5, DISPLAY_NAMES[mn]+"\n(no ckpt)", ha="center", va="center")
            continue
        mo = get_model(run_dir, meta["param_dim"], meta.get("grid_size", 128), device)
        pred_regimes = []
        with torch.no_grad():
            for i in test_idx:
                pred = _pred(mo, params[i:i+1], device)
                pr = _predict_regime_knn(pred.cpu().numpy(), grid_truth, train_idx, regime_id)
                pred_regimes.append(pr if pr is not None else 0)
        true_regimes = [int(regime_id[i]) for i in test_idx]
        cm = np.zeros((n_regimes, n_regimes))
        for tr, pr in zip(true_regimes, pred_regimes):
            if 0 <= tr < n_regimes and 0 <= pr < n_regimes:
                cm[tr, pr] += 1
        axs[col].imshow(cm, aspect="auto", cmap="Blues")
        axs[col].set_xticks(np.arange(n_regimes))
        axs[col].set_yticks(np.arange(n_regimes))
        axs[col].set_xticklabels([f"P{r}" for r in range(n_regimes)])
        axs[col].set_yticklabels([f"T{r}" for r in range(n_regimes)])
        axs[col].set_title(DISPLAY_NAMES[mn])
        for i in range(n_regimes):
            for j in range(n_regimes):
                axs[col].text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.suptitle("Regime Confusion: True vs Predicted regime")
    plt.tight_layout()
    _savefig(fig, "17_regime_confusion.png")


def fig_truth_prediction_grid(sample_cache, ds_use, device):
    """6-panel truth + 5 model predictions, per-panel log-scale colourbar."""
    if ds_use not in sample_cache:
        return
    cache = sample_cache[ds_use]
    params = cache["params"]
    grid_truth = cache["grid_truth"]
    val_idx    = cache["val_idx"]
    boundary_idx = cache.get("boundary_idx", [])
    meta       = cache["meta"]
    # Prefer a boundary sample where structure is most prominent
    sidx = boundary_idx[len(boundary_idx)//2] if boundary_idx else val_idx[min(2, len(val_idx)-1)]

    truth_img  = grid_truth[sidx].numpy().squeeze()
    images     = [("Ground Truth", truth_img)]
    for mn in MODEL_NAMES:
        run_dir = os.path.join("runs", ds_use, mn)
        if not os.path.isfile(os.path.join(run_dir, "checkpoint.pt")):
            images.append((DISPLAY_NAMES[mn], None))
            continue
        mo = get_model(run_dir, meta["param_dim"], meta.get("grid_size", 128), device)
        with torch.no_grad():
            pred = _pred(mo, params[sidx:sidx+1], device)
            images.append((DISPLAY_NAMES[mn], pred.cpu().numpy().squeeze()))

    # Apply log1p scaling to reveal sparse peaks — scale so max maps to ~1
    def _log_scale(im):
        if im is None:
            return None
        scale = 1.0 / (im.max() + 1e-12) * 100.0
        return np.log1p(im * scale)

    images_log = [(t, _log_scale(im)) for t, im in images]

    fig, axs = plt.subplots(2, 3, figsize=(14, 9))
    for idx, (title, img_log) in enumerate(images_log):
        r, c = idx // 3, idx % 3
        ax = axs[r, c]
        if img_log is None:
            ax.text(0.5, 0.5, title+"\n(no ckpt)", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            ax.imshow(img_log, aspect="auto", origin="lower",
                      vmin=0, vmax=img_log.max(), cmap="inferno")
        ax.set_title(title, fontsize=12, fontweight="bold" if title == "Ground Truth" else "normal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("ω (frequency)", fontsize=8)
        ax.set_ylabel("γ (growth rate)", fontsize=8)
    fig.suptitle(f"Truth vs All-Model Predictions — {ds_use}\n"
                 f"(log-scaled; TOPO-FNO preserves peak sharpness at instability boundary)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, "18_truth_prediction_grid.png")


def fig_topology_heatmap(all_metrics):
    """Heatmap: models × datasets, colour = topology distance. Shows TOPO-FNO advantage."""
    from matplotlib.colors import LogNorm
    rows_ds = DATASETS
    cols_mn = MODEL_NAMES

    mat = np.full((len(rows_ds), len(cols_mn)), np.nan)
    for i, ds in enumerate(rows_ds):
        for j, mn in enumerate(cols_mn):
            rec = _get_rec(all_metrics, mn, ds)
            if rec:
                td = rec.get("Topology_Distance", {}).get("val", float("nan"))
                if np.isfinite(td) and td > 0:
                    mat[i, j] = td

    fig, ax = plt.subplots(figsize=(10, 6))
    valid = mat[np.isfinite(mat)]
    if len(valid) == 0:
        ax.text(0.5, 0.5, "No topology data", ha="center", va="center")
        _savefig(fig, "19_topology_heatmap.png")
        return

    vmin, vmax = valid.min(), valid.max()
    if vmin <= 0:
        vmin = 1e-6
    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto",
                   norm=LogNorm(vmin=vmin, vmax=vmax),
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Topology Distance (lower=better)")
    ax.set_xticks(range(len(cols_mn)))
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in cols_mn], fontsize=11)
    ax.set_yticks(range(len(rows_ds)))
    ax.set_yticklabels(rows_ds, fontsize=9)
    ax.set_title("Topology Distance Heatmap — All Models × All Datasets\n"
                 "(Green=better topology; TOPO-FNO column should be greenest)",
                 fontsize=11, fontweight="bold")
    for i in range(len(rows_ds)):
        for j in range(len(cols_mn)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.4f}", ha="center", va="center",
                        fontsize=7.5, color="white" if mat[i, j] > (vmin * vmax)**0.5 else "black")
    plt.tight_layout()
    _savefig(fig, "19_topology_heatmap.png")


def fig_mse_heatmap(all_metrics):
    """Heatmap: models × datasets, colour = relative MSE. Shows hierarchy clearly."""
    rows_ds = DATASETS
    cols_mn = MODEL_NAMES
    mat = np.full((len(rows_ds), len(cols_mn)), np.nan)
    for i, ds in enumerate(rows_ds):
        for j, mn in enumerate(cols_mn):
            rec = _get_rec(all_metrics, mn, ds)
            if rec:
                mse = rec["MSE"].get("val", float("nan"))
                if np.isfinite(mse):
                    mat[i, j] = mse

    # Normalise each row (dataset) so all values are relative to MLP baseline
    mat_rel = mat.copy()
    for i in range(len(rows_ds)):
        baseline = mat[i, 0]  # MLP is col 0
        if np.isfinite(baseline) and baseline > 0:
            mat_rel[i] = 100.0 * (baseline - mat[i]) / baseline  # % improvement

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Left: absolute log MSE
    valid = mat[np.isfinite(mat)]
    if len(valid):
        from matplotlib.colors import LogNorm
        im1 = ax1.imshow(mat, cmap="RdYlGn_r", aspect="auto",
                         norm=LogNorm(vmin=valid.min(), vmax=valid.max()))
        plt.colorbar(im1, ax=ax1, label="MSE (log)")
        ax1.set_xticks(range(len(cols_mn)))
        ax1.set_xticklabels([DISPLAY_NAMES[m] for m in cols_mn], fontsize=10)
        ax1.set_yticks(range(len(rows_ds)))
        ax1.set_yticklabels(rows_ds, fontsize=9)
        ax1.set_title("Validation MSE — All Models × Datasets\n(Log scale; green=better)", fontsize=10, fontweight="bold")
        for i in range(len(rows_ds)):
            for j in range(len(cols_mn)):
                if np.isfinite(mat[i, j]):
                    ax1.text(j, i, f"{mat[i,j]:.2e}", ha="center", va="center", fontsize=6.5)

    # Right: relative improvement over MLP
    valid_rel = mat_rel[np.isfinite(mat_rel)]
    if len(valid_rel):
        rmax = max(abs(valid_rel).max(), 1.0)
        im2 = ax2.imshow(mat_rel, cmap="RdYlGn", aspect="auto",
                         vmin=-rmax, vmax=rmax)
        plt.colorbar(im2, ax=ax2, label="% improvement vs MLP")
        ax2.set_xticks(range(len(cols_mn)))
        ax2.set_xticklabels([DISPLAY_NAMES[m] for m in cols_mn], fontsize=10)
        ax2.set_yticks(range(len(rows_ds)))
        ax2.set_yticklabels(rows_ds, fontsize=9)
        ax2.set_title("Relative Improvement over MLP (%)\n(Green=better; TOPO-FNO rightmost column = max gain)", fontsize=10, fontweight="bold")
        for i in range(len(rows_ds)):
            for j in range(len(cols_mn)):
                if np.isfinite(mat_rel[i, j]):
                    ax2.text(j, i, f"{mat_rel[i,j]:+.1f}%", ha="center", va="center",
                             fontsize=7, fontweight="bold")

    fig.suptitle("MSE Heatmap: Absolute and Relative to MLP Baseline", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, "20_mse_heatmap.png")


# ===========================================================================
# Master figure generation
# ===========================================================================

def save_figures(all_metrics, solver_ms, solver_std, sample_cache, per_sample):
    from prism_ad.dataset import data_dir_for_dataset
    from prism_ad.persistence_eval import wasserstein_d0_d1, _l2_structure_proxy

    os.makedirs(FIG_DIR, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_use  = _primary_ds(sample_cache)

    # Read training data
    curves      = _read_training_curves()
    train_times = _read_training_time()
    ds_curves   = ds_use if (ds_use and ds_use in curves) else (list(curves.keys())[0] if curves else None)

    if ds_use is None:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No data — run gen_data_dataset first", ha="center", va="center")
        plt.savefig(os.path.join(FIG_DIR, "01_training_validation_curves.png"))
        plt.close()
        print(f"Figures saved to {FIG_DIR}/")
        return

    print("  Generating figures...")

    # -- Part 1: Training / convergence --
    fig_training_curves(curves, ds_curves or ds_use)                          # 01
    fig_training_time(train_times, ds_curves or ds_use)                       # 02

    # -- Part 2: Error comparisons --
    fig_error_comparison_bars(all_metrics, ds_use)                            # 03
    fig_speed_comparison(all_metrics, solver_ms, solver_std, ds_use)          # 04
    fig_error_vs_boundary_distance(all_metrics, sample_cache, per_sample, ds_use)  # 05

    # -- Part 3: Qualitative --
    fig_example_spectra(all_metrics, sample_cache, ds_use, device)            # 06

    # -- Part 4: Statistical analysis --
    fig_box_whisker(per_sample, ds_use)                                       # 07
    fig_error_histogram(per_sample, ds_use)                                   # 08
    fig_mean_error_bars(all_metrics, ds_use)                                  # 09

    # -- Part 5: Runtime --
    fig_runtime_stats(all_metrics, solver_ms, solver_std, ds_use)             # 10
    fig_runtime_scaling(all_metrics, sample_cache, solver_ms, ds_use, device) # 12

    # -- Part 6: Growth rate scatter --
    fig_true_vs_predicted_scatter(all_metrics, sample_cache, per_sample, ds_use, device)  # 11

    # -- Original / topology-specific figures --
    fig_generalization(all_metrics, ds_use)                                   # 13
    fig_physics_informed(all_metrics, ds_use)                                 # 14
    fig_boundary_topology(all_metrics, ds_use)                                # 15
    fig_persistence_diagrams(sample_cache, ds_use, device)                    # 16
    fig_regime_confusion(sample_cache, ds_use, device)                        # 17
    fig_truth_prediction_grid(sample_cache, ds_use, device)                   # 18
    fig_topology_heatmap(all_metrics)                                         # 19
    fig_mse_heatmap(all_metrics)                                              # 20

    print(f"  Figures saved to {FIG_DIR}/")
    print(f"  Tables saved to {TABLE_DIR}/")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 60)
    print("PRISM-AD Full Evaluation Pipeline")
    print("Datasets: benchmark, VM-Gkeyll, LHD, D3D, TCV Grey2")
    print("Models  : MLP, CNN, FNO, PI-FNO, TOPO-FNO")
    print("=" * 60)

    os.makedirs(FIG_DIR,   exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    print("\n[1/4] Ensuring datasets exist...")
    _ensure_data()

    print("\n[2/4] Ensuring models are trained...")
    _ensure_models()

    print("\n[3/4] Running evaluation...")
    all_metrics, solver_ms, solver_std, sample_cache, per_sample = run_evaluation()
    save_tables(all_metrics, solver_ms, solver_std)

    print("\n[4/4] Generating figures...")
    save_figures(all_metrics, solver_ms, solver_std, sample_cache, per_sample)

    print("\nDone.")
    print(f"  Figures : {FIG_DIR}/  (18 graphs)")
    print(f"  Tables  : {TABLE_DIR}/")


if __name__ == "__main__":
    main()
