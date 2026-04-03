# Eval all models on all datasets; write metrics_table.csv per dataset and all_datasets_metrics.csv.

import argparse
import csv
import glob
import json
import os
import numpy as np
import torch
import torch.nn.functional as F

from prism_ad.models import MLP, CNN, FNO2d, PIFNO, TopoFNO
from prism_ad.dataset import load_dataset, load_meta, data_dir_for_dataset
from prism_ad.persistence_eval import topology_distance_batch
from prism_ad.training import gamma_max_from_density


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
    p.add_argument("--datasets", nargs="+", default=["benchmark_operator", "vlasov_maxwell_gkeyll", "fusion_real_lhd"])
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_names = ["mlp", "cnn", "fno", "pi_fno", "topo_fno"]
    all_rows = []

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
        gamma_max = data["gamma_max"]
        val_idx = splits["val_idx"]
        boundary_idx = splits.get("boundary_idx") or []
        ood_params = data.get("ood_params")
        grid_ood = data.get("grid_ood")
        has_gamma_max = meta.get("has_gamma_max", True)
        run_base = os.path.join("runs", dataset_name)
        results_dir = os.path.join("runs", "results", dataset_name)
        os.makedirs(results_dir, exist_ok=True)

        table_rows = []
        for model_name in model_names:
            run_dir = os.path.join(run_base, model_name)
            if not os.path.isfile(os.path.join(run_dir, "checkpoint.pt")):
                continue
            model = get_model(run_dir, param_dim, grid_size, device)
            with torch.no_grad():
                val_mse = val_n = 0.0
                val_preds = []
                for i in val_idx:
                    pred = model(params[i : i + 1].to(device))
                    if hasattr(model, "stab_head") and isinstance(pred, tuple):
                        pred = pred[0]
                    val_mse += F.mse_loss(pred, grid_truth[i : i + 1].to(device)).item()
                    val_n += 1
                    val_preds.append(pred.cpu().numpy())
                val_mse /= max(val_n, 1)
                val_preds = np.concatenate(val_preds, 0)
                val_truths = grid_truth[val_idx].numpy()
                w0, w1, topo, sp, n0, n1 = topology_distance_batch(val_preds, val_truths)
                boundary_mse = 0.0
                if boundary_idx:
                    for i in boundary_idx:
                        if i < params.shape[0]:
                            pred = model(params[i : i + 1].to(device))
                            if hasattr(model, "stab_head") and isinstance(pred, tuple):
                                pred = pred[0]
                            boundary_mse += F.mse_loss(pred, grid_truth[i : i + 1].to(device)).item()
                    boundary_mse /= max(len(boundary_idx), 1)
                ood_mse = float("nan")
                if ood_params is not None and grid_ood is not None:
                    n_ood = min(50, grid_ood.shape[0])
                    ood_mse = 0.0
                    for i in range(n_ood):
                        pred = model(ood_params[i : i + 1].to(device))
                        if hasattr(model, "stab_head") and isinstance(pred, tuple):
                            pred = pred[0]
                        ood_mse += F.mse_loss(pred, grid_ood[i : i + 1].to(device)).item()
                    ood_mse /= n_ood
                gamma_max_err = float("nan")
                if has_gamma_max and meta.get("gamma_min") is not None and meta.get("gamma_max") is not None:
                    pred_t = torch.from_numpy(val_preds[:20]).to(device)
                    g_pred = gamma_max_from_density(pred_t, meta["gamma_min"], meta["gamma_max"], device)
                    g_true = gamma_max[val_idx[:20]].to(device)
                    gamma_max_err = F.l1_loss(g_pred, g_true).item()

            row = {
                "dataset": dataset_name,
                "model": model_name,
                "val_mse": val_mse,
                "boundary_mse": boundary_mse,
                "ood_mse": ood_mse,
                "gamma_max_err": gamma_max_err,
                "W0": w0 if not np.isnan(w0) else None,
                "W1": w1 if not np.isnan(w1) else None,
                "TopoDist": float(topo) if topo is not None else None,
                "structure_proxy_distance": sp,
            }
            table_rows.append(row)
            all_rows.append(row)
            torch.save({"val_preds": val_preds, "val_truths": val_truths}, os.path.join(run_dir, "eval_preds.pt"))
            with open(os.path.join(run_dir, "eval_metrics.json"), "w") as f:
                json.dump(row, f, indent=2)

        if table_rows:
            with open(os.path.join(results_dir, "metrics_table.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["dataset", "model", "val_mse", "boundary_mse", "ood_mse", "gamma_max_err", "W0", "W1", "TopoDist", "structure_proxy_distance"])
                w.writeheader()
                w.writerows(table_rows)

        if dataset_name == "benchmark_operator" and "regime_id" in data:
            regime_id = data["regime_id"].numpy()
            from collections import Counter
            counts = Counter(regime_id)
            with open(os.path.join(results_dir, "regimes_table.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["regime_id", "count"])
                for r in sorted(counts.keys()):
                    w.writerow([int(r), counts[r]])

    if all_rows:
        os.makedirs("runs/results", exist_ok=True)
        with open("runs/results/all_datasets_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "model", "val_mse", "boundary_mse", "ood_mse", "gamma_max_err", "W0", "W1", "TopoDist", "structure_proxy_distance"])
            w.writeheader()
            w.writerows(all_rows)
    print("eval_all_datasets done", len(all_rows), "rows")


if __name__ == "__main__":
    main()
