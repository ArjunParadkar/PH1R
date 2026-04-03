# Generate 10–15+ figures per dataset under runs/figs/<dataset_name>/.

import argparse
import csv
import glob
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism_ad.dataset import load_dataset, load_meta, data_dir_for_dataset
from prism_ad.models import MLP, CNN, FNO2d, PIFNO, TopoFNO
from prism_ad.persistence_eval import wasserstein_d0_d1, get_diagrams


def get_model(run_dir, param_dim, grid_size, device):
    name = os.path.basename(run_dir.rstrip("/"))
    models = {"mlp": MLP, "cnn": CNN, "fno": FNO2d, "pi_fno": PIFNO, "topo_fno": TopoFNO}
    if name not in models:
        return None
    model = models[name](param_dim, grid_size, 16, 16, 4) if name in ("fno", "pi_fno", "topo_fno") else models[name](param_dim, grid_size)
    ckpt = os.path.join(run_dir, "checkpoint.pt")
    if os.path.isfile(ckpt):
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state))
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["benchmark_operator", "vlasov_maxwell_gkeyll", "fusion_real_lhd"])
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_names = ["mlp", "cnn", "fno", "pi_fno", "topo_fno"]

    for dataset_name in args.datasets:
        data_dir = data_dir_for_dataset(dataset_name)
        if not os.path.isfile(os.path.join(data_dir, "dataset.pt")):
            continue
        data, splits = load_dataset(data_dir)
        meta = load_meta(data_dir)
        param_dim = meta["param_dim"]
        grid_size = meta.get("grid_size", 128)
        params = data["params"]
        grid_truth = data["grid_truth"]
        gamma_max = data["gamma_max"].numpy()
        val_idx = splits["val_idx"]
        run_base = os.path.join("runs", dataset_name)
        fig_dir = os.path.join("runs", "figs", dataset_name)
        os.makedirs(fig_dir, exist_ok=True)
        run_dirs = [os.path.join(run_base, n) for n in model_names]
        run_dirs = [d for d in run_dirs if os.path.isfile(os.path.join(d, "checkpoint.pt"))]

        sample_val_idx = min(2, len(val_idx) - 1)
        sample_idx = val_idx[sample_val_idx]
        truth_img = grid_truth[sample_idx].numpy().squeeze()
        extent = [meta.get("omega_min", -0.5), meta.get("omega_max", 0.5), meta.get("gamma_min", -0.3), meta.get("gamma_max", 0.3)] if meta.get("omega_min") is not None else None

        plt.figure(figsize=(5, 4))
        plt.imshow(truth_img, aspect="auto", origin="lower", extent=extent if extent else [0, 1, 0, 1])
        plt.colorbar(label="density")
        plt.xlabel("$\\omega$" if extent else "x")
        plt.ylabel("$\\gamma$" if extent else "y")
        plt.title(f"{dataset_name}: density (128×128)")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "01_density_mapping.png"))
        plt.close()

        if "eigs" in data:
            eigs_sample = np.asarray(data["eigs"][sample_idx]).flatten()
            eigs_sample = eigs_sample[np.isfinite(eigs_sample)]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            ax1.scatter(eigs_sample.real, eigs_sample.imag, alpha=0.7, s=20)
            if extent:
                ax1.set_xlim(extent[0], extent[1])
                ax1.set_ylim(extent[2], extent[3])
            ax1.set_xlabel("$\\omega$")
            ax1.set_ylabel("$\\gamma$")
            ax1.set_title("Eigenvalue cloud")
            ax2.imshow(truth_img, aspect="auto", origin="lower", extent=extent or [0, 1, 0, 1])
            ax2.set_title("Density grid")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "02_cloud_to_density.png"))
            plt.close()

        if "regime_id" in data and dataset_name == "benchmark_operator":
            regimes = data["regime_id"].numpy()
            fig, axs = plt.subplots(2, 2, figsize=(8, 8))
            for r in range(4):
                idx = np.where(regimes == r)[0]
                if len(idx):
                    axs[r // 2, r % 2].imshow(grid_truth[idx[0]].numpy().squeeze(), aspect="auto", origin="lower")
                    axs[r // 2, r % 2].set_title(f"Regime {r}")
            plt.suptitle("Regime gallery")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "03_regime_gallery.png"))
            plt.close()

        plt.figure(figsize=(6, 3))
        plt.hist(gamma_max, bins=20, edgecolor="black", alpha=0.7)
        plt.axvline(0, color="red", linestyle="--", label="stability")
        plt.xlabel("$\\gamma_{max}$")
        plt.ylabel("count")
        plt.legend()
        plt.title("Stability boundary")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "04_stability_boundary_hist.png"))
        plt.close()

        n_models = len(run_dirs)
        if n_models:
            fig, axs = plt.subplots(2, n_models + 1, figsize=(3 * (n_models + 1), 6))
            axs[0, 0].imshow(truth_img, aspect="auto", origin="lower")
            axs[0, 0].set_title("Truth")
            for k, run_dir in enumerate(run_dirs):
                name = os.path.basename(run_dir)
                model = get_model(run_dir, param_dim, grid_size, device)
                if model is None:
                    continue
                pred_path = os.path.join(run_dir, "eval_preds.pt")
                if os.path.isfile(pred_path):
                    ev = torch.load(pred_path, map_location="cpu", weights_only=True)
                    pred_img = ev["val_preds"][sample_val_idx].squeeze()
                else:
                    with torch.no_grad():
                        pred_img = model(params[sample_idx : sample_idx + 1].to(device))
                        if hasattr(model, "stab_head") and isinstance(pred_img, tuple):
                            pred_img = pred_img[0]
                        pred_img = pred_img.cpu().numpy().squeeze()
                axs[0, k + 1].imshow(pred_img, aspect="auto", origin="lower")
                axs[0, k + 1].set_title(name)
                axs[1, k + 1].imshow(np.abs(pred_img - truth_img), aspect="auto", origin="lower")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "05_truth_pred_diff_grid.png"))
            plt.close()

        metrics_path = os.path.join("runs", "results", dataset_name, "metrics_table.csv")
        if os.path.isfile(metrics_path):
            with open(metrics_path) as f:
                rows = list(csv.DictReader(f))
            if rows:
                models_in = list({r["model"] for r in rows})
                metrics_in = ["val_mse", "boundary_mse", "TopoDist", "structure_proxy_distance"]
                M = np.zeros((len(models_in), len(metrics_in)))
                for i, m in enumerate(models_in):
                    r = next(x for x in rows if x["model"] == m)
                    for j, key in enumerate(metrics_in):
                        val = r.get(key)
                        M[i, j] = float(val) if val and val != "" else np.nan
                fig, ax = plt.subplots(figsize=(6, 4))
                im = ax.imshow(np.nan_to_num(M, nan=0), aspect="auto")
                ax.set_xticks(range(len(metrics_in)))
                ax.set_xticklabels(metrics_in)
                ax.set_yticks(range(len(models_in)))
                ax.set_yticklabels(models_in)
                plt.colorbar(im, ax=ax)
                plt.title("Model comparison heatmap")
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, "06_model_comparison_heatmap.png"))
                plt.close()

                w0s = [float(r.get("W0") or 0) for r in rows if r.get("model") in models_in]
                w1s = [float(r.get("W1") or 0) for r in rows if r.get("model") in models_in]
                td = [float(r.get("TopoDist") or r.get("structure_proxy_distance") or 0) for r in rows if r.get("model") in models_in]
                x = np.arange(len(models_in))
                width = 0.25
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(x - width, w0s[:len(models_in)], width, label="W0")
                ax.bar(x, w1s[:len(models_in)], width, label="W1")
                ax.bar(x + width, td[:len(models_in)], width, label="TopoDist/proxy")
                ax.set_xticks(x)
                ax.set_xticklabels(models_in)
                ax.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, "07_wasserstein_barchart.png"))
                plt.close()

        boundary_errors = {n: [] for n in model_names}
        gamma_abs_list = []
        for idx in val_idx[:60]:
            gamma_abs_list.append(abs(gamma_max[idx]))
            for run_dir in run_dirs:
                name = os.path.basename(run_dir)
                model = get_model(run_dir, param_dim, grid_size, device)
                if model is None:
                    continue
                with torch.no_grad():
                    pred = model(params[idx : idx + 1].to(device))
                    if hasattr(model, "stab_head") and isinstance(pred, tuple):
                        pred = pred[0]
                    err = F.mse_loss(pred, grid_truth[idx : idx + 1].to(device)).item()
                boundary_errors[name].append(err)
        if gamma_abs_list and any(boundary_errors.values()):
            plt.figure(figsize=(6, 4))
            for name in model_names:
                if boundary_errors[name]:
                    plt.scatter(gamma_abs_list, boundary_errors[name], label=name, alpha=0.6)
            plt.xlabel("|$\\gamma_{max}$|")
            plt.ylabel("MSE")
            plt.legend()
            plt.title("Boundary stress")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "08_boundary_stress.png"))
            plt.close()

        run_dir_topo = os.path.join(run_base, "topo_fno")
        if os.path.isfile(os.path.join(run_dir_topo, "eval_preds.pt")):
            ev = torch.load(os.path.join(run_dir_topo, "eval_preds.pt"), map_location="cpu", weights_only=True)
            preds = ev["val_preds"]
            truths = ev["val_truths"]
            idx = min(3, preds.shape[0] - 1)
            d0_t, d1_t = get_diagrams(truths[idx])
            d0_p, d1_p = get_diagrams(preds[idx])
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            if d0_t is not None and d0_t.size > 0:
                ax[0].scatter(d0_t[:, 0], d0_t[:, 1], label="truth H0")
            if d0_p is not None and d0_p.size > 0:
                ax[0].scatter(d0_p[:, 0], d0_p[:, 1], label="pred H0")
            ax[0].set_title("H0")
            ax[0].legend()
            if d1_t is not None and d1_t.size > 0:
                ax[1].scatter(d1_t[:, 0], d1_t[:, 1], label="truth H1")
            if d1_p is not None and d1_p.size > 0:
                ax[1].scatter(d1_p[:, 0], d1_p[:, 1], label="pred H1")
            ax[1].set_title("H1")
            ax[1].legend()
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "09_persistence_diagrams.png"))
            plt.close()

        for run_dir in run_dirs:
            curve = os.path.join(run_dir, "train_curve.png")
            if os.path.isfile(curve):
                import shutil
                name = os.path.basename(run_dir)
                shutil.copy(curve, os.path.join(fig_dir, f"10_train_curve_{name}.png"))
                break

        manifest_path = os.path.join(data_dir, "manifest.csv")
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                rows = list(csv.DictReader(f))[:10]
            if rows:
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.axis("off")
                cols = list(rows[0].keys())
                cell_text = [[r.get(c, "") for c in cols] for r in rows]
                ax.table(cellText=cell_text, colLabels=cols, loc="center", cellLoc="left")
                plt.title("Dataset preview (first 10 rows)")
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, "11_manifest_preview.png"))
                plt.close()

    print("make_figs_dataset done")


if __name__ == "__main__":
    main()
