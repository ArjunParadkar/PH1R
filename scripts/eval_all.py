# Evaluate all runs: val/boundary/OOD MSE and topology distance; write metrics to each run and combined.

import argparse
import glob
import json
import os
import torch
import torch.nn.functional as F

from prism_ad.models import MLP, CNN, FNO2d, PIFNO, TopoFNO
from prism_ad.dataset import load_dataset, load_meta
import numpy as np
from prism_ad.persistence_eval import topology_distance_batch


def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


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
        if "model" in state:
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


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
    gamma_max = data["gamma_max"]
    ood_params = data["ood_params"]
    grid_ood = data["grid_ood"]
    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    boundary_idx = splits["boundary_idx"]

    all_metrics = {}
    for run_dir in run_dirs:
        model = get_model(run_dir, param_dim, grid_size, device)
        if model is None:
            continue
        name = os.path.basename(run_dir.rstrip("/"))
        with torch.no_grad():
            val_mse = 0.0
            val_n = 0
            val_preds = []
            val_truths = []
            for i in val_idx:
                p_b = params[i : i + 1].to(device)
                t_b = grid_truth[i : i + 1].to(device)
                pred = model(p_b) if not hasattr(model, "stab_head") else model(p_b, return_stab=False)
                val_mse += F.mse_loss(pred, t_b).item()
                val_n += 1
                val_preds.append(pred.cpu().numpy())
                val_truths.append(t_b.cpu().numpy())
            val_mse /= max(val_n, 1)
            val_preds = np.concatenate(val_preds, 0)
            val_truths = np.concatenate(val_truths, 0)
            w0, w1, topo, structure_proxy, num_H0, num_H1 = topology_distance_batch(val_preds, val_truths)
            boundary_mse = 0.0
            bound_n = 0
            if boundary_idx:
                for i in boundary_idx:
                    if i >= params.shape[0]:
                        continue
                    p_b = params[i : i + 1].to(device)
                    t_b = grid_truth[i : i + 1].to(device)
                    pred = model(p_b) if not hasattr(model, "stab_head") else model(p_b, return_stab=False)
                    boundary_mse += F.mse_loss(pred, t_b).item()
                    bound_n += 1
                boundary_mse /= max(bound_n, 1)
            ood_mse = 0.0
            ood_n = min(grid_ood.shape[0], 100)
            for i in range(ood_n):
                p_b = ood_params[i : i + 1].to(device)
                t_b = grid_ood[i : i + 1].to(device)
                pred = model(p_b) if not hasattr(model, "stab_head") else model(p_b, return_stab=False)
                ood_mse += F.mse_loss(pred, t_b).item()
            ood_mse /= max(ood_n, 1)

        metrics = {
            "val_mse": val_mse,
            "boundary_mse": boundary_mse,
            "ood_mse": ood_mse,
            "topo_w0": float(w0) if not np.isnan(w0) else None,
            "topo_w1": float(w1) if not np.isnan(w1) else None,
            "topo_dist": float(topo) if topo is not None and not np.isnan(topo) else None,
            "structure_proxy_distance": float(structure_proxy),
            "num_H0_features": num_H0,
            "num_H1_features": num_H1,
        }
        all_metrics[name] = metrics
        with open(os.path.join(run_dir, "eval_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        torch.save({"val_preds": val_preds, "val_truths": val_truths}, os.path.join(run_dir, "eval_preds.pt"))

    os.makedirs("runs", exist_ok=True)
    with open("runs/eval_all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("eval_all done", list(all_metrics.keys()))


if __name__ == "__main__":
    main()
