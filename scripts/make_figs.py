# Produce all required figures under runs/<name>/figs and runs/figs.

import argparse
import glob
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism_ad.dataset import load_dataset, load_meta
from prism_ad.models import MLP, CNN, FNO2d, PIFNO, TopoFNO
from prism_ad.persistence_eval import wasserstein_d0_d1


def get_model(run_dir, param_dim, grid_size, device):
    name = os.path.basename(run_dir.rstrip("/"))
    if name == "mlp":
        model = MLP(param_dim, grid_size)
    elif name == "cnn":
        model = CNN(param_dim, grid_size)
    elif name == "fno":
        model = FNO2d(param_dim, grid_size, 16, 16, 4)
    elif name == "pi_fno":
        model = PIFNO(param_dim, grid_size, 16, 16, 4)
    elif name == "topo_fno":
        model = TopoFNO(param_dim, grid_size, 16, 16, 4)
    else:
        return None
    ckpt = os.path.join(run_dir, "checkpoint.pt")
    if os.path.isfile(ckpt):
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state))
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/*")
    args = p.parse_args()
    run_dirs = sorted(glob.glob(args.runs))
    run_dirs = [d for d in run_dirs if os.path.isdir(d) and os.path.isfile(os.path.join(d, "checkpoint.pt"))]
    meta = load_meta("data/processed")
    data, splits = load_dataset("data/processed")
    param_dim = meta["param_dim"]
    grid_size = meta["grid_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = data["params"]
    grid_truth = data["grid_truth"]
    gamma_max = data["gamma_max"].numpy()
    val_idx = splits["val_idx"]

    os.makedirs("runs/figs", exist_ok=True)

    sample_val_idx = min(2, len(val_idx) - 1)
    sample_idx = val_idx[sample_val_idx]
    truth_img = grid_truth[sample_idx].numpy().squeeze()
    plt.figure(figsize=(5, 4))
    plt.imshow(truth_img, aspect="auto", origin="lower", extent=[meta["omega_min"], meta["omega_max"], meta["gamma_min"], meta["gamma_max"]])
    plt.colorbar(label="density")
    plt.xlabel("$\\omega$")
    plt.ylabel("$\\gamma$")
    plt.title("Eigenvalue density (128×128)")
    plt.tight_layout()
    plt.savefig("runs/figs/density_mapping.png")
    plt.close()

    if "eigs" in data:
        eigs_sample = np.asarray(data["eigs"][sample_idx]).flatten()
        eigs_sample = eigs_sample[np.isfinite(eigs_sample)]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.scatter(eigs_sample.real, eigs_sample.imag, alpha=0.7, s=20)
        ax1.set_xlim(meta["omega_min"], meta["omega_max"])
        ax1.set_ylim(meta["gamma_min"], meta["gamma_max"])
        ax1.set_xlabel("$\\omega$")
        ax1.set_ylabel("$\\gamma$")
        ax1.set_title("Eigenvalue cloud (scatter)")
        ax2.imshow(truth_img, aspect="auto", origin="lower", extent=[meta["omega_min"], meta["omega_max"], meta["gamma_min"], meta["gamma_max"]])
        ax2.set_xlabel("$\\omega$")
        ax2.set_ylabel("$\\gamma$")
        ax2.set_title("Density grid (128×128)")
        plt.tight_layout()
        plt.savefig("runs/figs/eigenvalue_cloud_to_density.png")
        plt.close()

    model_names = [os.path.basename(d.rstrip("/")) for d in run_dirs]
    n_models = len(model_names)
    fig, axs = plt.subplots(2, n_models + 1, figsize=(3 * (n_models + 1), 6))
    if n_models == 0:
        axs = np.array([[plt.gca()], [plt.gca()]])
    axs[0, 0].imshow(truth_img, aspect="auto", origin="lower")
    axs[0, 0].set_title("Truth")
    axs[1, 0].axis("off")
    for k, (run_dir, name) in enumerate(zip(run_dirs, model_names)):
        model = get_model(run_dir, param_dim, grid_size, device)
        if model is None:
            continue
        pred_path = os.path.join(run_dir, "eval_preds.pt")
        if os.path.isfile(pred_path):
            ev = torch.load(pred_path, map_location="cpu", weights_only=True)
            preds = ev["val_preds"]
            i = min(sample_val_idx, preds.shape[0] - 1)
            pred_img = preds[i].squeeze()
        else:
            with torch.no_grad():
                pred_img = model(params[sample_idx : sample_idx + 1].to(device))
                if hasattr(model, "stab_head"):
                    pred_img = pred_img[0] if isinstance(pred_img, tuple) else pred_img
                pred_img = pred_img.cpu().numpy().squeeze()
        axs[0, k + 1].imshow(pred_img, aspect="auto", origin="lower")
        axs[0, k + 1].set_title(name)
        axs[1, k + 1].imshow(np.abs(pred_img - truth_img), aspect="auto", origin="lower")
        axs[1, k + 1].set_title(f"|pred-truth|")
    plt.tight_layout()
    plt.savefig("runs/figs/model_comparison_grid.png")
    plt.close()

    eval_path = "runs/eval_all_metrics.json"
    if os.path.isfile(eval_path):
        with open(eval_path) as f:
            all_m = json.load(f)
        names = list(all_m.keys())
        val_mses = [all_m[n].get("val_mse", 0) for n in names]
        topo_vals = []
        for n in names:
            m = all_m[n]
            topo_vals.append(m.get("topo_dist") if m.get("topo_dist") is not None else m.get("structure_proxy_distance", 0))
        use_topo_label = any(all_m[n].get("topo_dist") is not None for n in names)
        dist_label = "TopoDist" if use_topo_label else "Structure proxy dist"
        x = np.arange(len(names))
        width = 0.2
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x - 1.5 * width, val_mses, width, label="Val MSE")
        w0s = [all_m[n].get("topo_w0") or 0 for n in names]
        w1s = [all_m[n].get("topo_w1") or 0 for n in names]
        ax.bar(x - 0.5 * width, w0s, width, label="W0")
        ax.bar(x + 0.5 * width, w1s, width, label="W1")
        ax.bar(x + 1.5 * width, topo_vals, width, label=dist_label)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()
        ax.set_title("Val MSE, W0, W1, " + dist_label + " by model")
        plt.tight_layout()
        plt.savefig("runs/figs/ph_distance_barchart.png")
        plt.close()

        def _topo_val(m):
            return m.get("topo_dist") if m.get("topo_dist") is not None else m.get("structure_proxy_distance") or 0
        fno_mse = all_m.get("fno", {}).get("val_mse")
        topo_mse = all_m.get("topo_fno", {}).get("val_mse")
        fno_topo = _topo_val(all_m.get("fno", {}))
        topo_topo = _topo_val(all_m.get("topo_fno", {}))
        pi_topo_v = _topo_val(all_m.get("pi_fno", {}))
        deltas = [
            ("PI-FNO vs FNO", all_m.get("pi_fno", {}).get("val_mse", 0) - (fno_mse or 0), pi_topo_v - fno_topo),
            ("Topo-FNO vs PI-FNO", (topo_mse or 0) - (all_m.get("pi_fno", {}).get("val_mse", 0)), topo_topo - pi_topo_v),
            ("Topo-FNO vs FNO", (topo_mse or 0) - (fno_mse or 0), topo_topo - fno_topo),
        ]
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(3)
        ax.bar(x - 0.2, [d[1] for d in deltas], 0.4, label="$\\Delta$ MSE")
        ax.bar(x + 0.2, [d[2] for d in deltas], 0.4, label="$\\Delta$ " + dist_label)
        ax.set_xticks(x)
        ax.set_xticklabels([d[0] for d in deltas])
        ax.legend()
        ax.axhline(0, color="gray", linestyle="--")
        plt.tight_layout()
        plt.savefig("runs/figs/improvement_bars.png")
        plt.close()

        ood_mses = [all_m.get(n, {}).get("ood_mse", 0) for n in names]
        plt.figure(figsize=(6, 4))
        plt.bar(names, ood_mses)
        plt.ylabel("OOD MSE")
        plt.title("OOD error (shift factor from config)")
        plt.tight_layout()
        plt.savefig("runs/figs/ood_plot.png")
        plt.close()

    boundary_errors = {name: [] for name in model_names}
    gamma_abs = []
    models_cache = {}
    for run_dir, name in zip(run_dirs, model_names):
        models_cache[name] = get_model(run_dir, param_dim, grid_size, device)
    for idx in val_idx[:80]:
        g = abs(gamma_max[idx])
        gamma_abs.append(g)
        for name in model_names:
            model = models_cache.get(name)
            if model is None:
                continue
            with torch.no_grad():
                pred = model(params[idx : idx + 1].to(device))
                if hasattr(model, "stab_head") and isinstance(pred, tuple):
                    pred = pred[0]
                err = F.mse_loss(pred, grid_truth[idx : idx + 1].to(device)).item()
            boundary_errors[name].append(err)
    if gamma_abs and any(boundary_errors.values()):
        fig, ax = plt.subplots(figsize=(6, 4))
        for name in model_names:
            if boundary_errors[name]:
                ax.scatter(gamma_abs, boundary_errors[name], label=name, alpha=0.6)
        ax.set_xlabel("|$\\gamma_{max}$|")
        ax.set_ylabel("MSE")
        ax.legend()
        ax.set_title("Boundary stress: error vs |gamma_max|")
        plt.tight_layout()
        plt.savefig("runs/figs/boundary_stress.png")
        plt.close()

    for run_dir in run_dirs:
        os.makedirs(os.path.join(run_dir, "figs"), exist_ok=True)
        name = os.path.basename(run_dir.rstrip("/"))
        pred_path = os.path.join(run_dir, "eval_preds.pt")
        if os.path.isfile(pred_path):
            ev = torch.load(pred_path, map_location="cpu", weights_only=True)
            preds = ev["val_preds"]
            truths = ev["val_truths"]
            idx = min(5, preds.shape[0] - 1)
            from prism_ad.persistence_eval import get_diagrams
            d0_t, d1_t = get_diagrams(truths[idx])
            d0_p, d1_p = get_diagrams(preds[idx])
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            if d0_t is not None and d0_t.size > 0:
                ax[0].scatter(d0_t[:, 0], d0_t[:, 1], label="truth H0", alpha=0.7)
            if d0_p is not None and d0_p.size > 0:
                ax[0].scatter(d0_p[:, 0], d0_p[:, 1], label="pred H0", alpha=0.7)
            ax[0].set_xlabel("birth")
            ax[0].set_ylabel("death")
            ax[0].set_title("H0")
            ax[0].legend()
            if d1_t is not None and d1_t.size > 0:
                ax[1].scatter(d1_t[:, 0], d1_t[:, 1], label="truth H1", alpha=0.7)
            if d1_p is not None and d1_p.size > 0:
                ax[1].scatter(d1_p[:, 0], d1_p[:, 1], label="pred H1", alpha=0.7)
            ax[1].set_xlabel("birth")
            ax[1].set_ylabel("death")
            ax[1].set_title("H1")
            ax[1].legend()
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "figs", "persistence_diagrams.png"))
            plt.close()

    if os.path.isfile(eval_path) and "fno" in all_m:
        fno_pred_path = None
        for d in run_dirs:
            if os.path.basename(d.rstrip("/")) == "fno":
                fno_pred_path = os.path.join(d, "eval_preds.pt")
                break
        if fno_pred_path and os.path.isfile(fno_pred_path):
            ev = torch.load(fno_pred_path, map_location="cpu", weights_only=True)
            preds_fno = ev["val_preds"]
            truths_fno = ev["val_truths"]
            mse_per = np.mean((preds_fno - truths_fno) ** 2, axis=(1, 2, 3))
            from prism_ad.persistence_eval import _l2_structure_proxy
            topo_per = []
            for i in range(min(50, preds_fno.shape[0])):
                _, _, td = wasserstein_d0_d1(preds_fno[i], truths_fno[i])
                if not np.isnan(td):
                    topo_per.append(td)
                else:
                    topo_per.append(_l2_structure_proxy(preds_fno[i], truths_fno[i]))
            L = min(len(topo_per), len(mse_per))
            topo_per = np.array(topo_per[:L], dtype=np.float64)
            mse_slice = np.array(mse_per[:L])
            np.nan_to_num(topo_per, nan=0.0, copy=False)
            median_mse = np.median(mse_slice)
            p90_topo = np.percentile(topo_per, 90)
            candidates = np.where((mse_slice < median_mse) & (topo_per > p90_topo))[0]
            if len(candidates) < 3:
                ratio = topo_per / (mse_slice + 1e-10)
                candidates = np.argsort(ratio)[-3:]
            else:
                candidates = candidates[np.argsort(topo_per[candidates])[-3:]]
            fig, axs = plt.subplots(2, 3, figsize=(9, 6))
            for col, idx in enumerate(candidates):
                idx = int(idx)
                axs[0, col].imshow(truths_fno[idx].squeeze(), aspect="auto", origin="lower")
                axs[0, col].set_title(f"Truth (MSE={mse_slice[idx]:.2e})")
                axs[1, col].imshow(preds_fno[idx].squeeze(), aspect="auto", origin="lower")
                axs[1, col].set_title(f"FNO pred (Topo={topo_per[min(idx, len(topo_per)-1)]:.2e})")
            plt.suptitle("MSE Lies: low MSE but high TopoDist")
            plt.tight_layout()
            plt.savefig("runs/figs/mse_lies_panel.png")
            plt.close()

    if os.path.isfile("runs/runtime.json"):
        with open("runs/runtime.json") as f:
            rt = json.load(f)
        solver_ms = rt.get("solver_ms_per_sample") or 1.0
        model_times = [(k, v) for k, v in rt.items() if k.endswith("_ms") and k != "solver_ms_per_sample"]
        fig, ax = plt.subplots(figsize=(6, 4))
        names = [k.replace("_ms", "") for k, _ in model_times]
        times = [t for _, t in model_times]
        speedups = [solver_ms / t for t in times]
        ax.barh(names, speedups)
        ax.set_xscale("log")
        ax.set_xlabel("Speedup (solver / model)")
        ax.set_title("Runtime speedup")
        plt.tight_layout()
        plt.savefig("runs/figs/runtime_speedup.png")
        plt.close()

    print("make_figs done", "runs/figs/")


if __name__ == "__main__":
    main()
