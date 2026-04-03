#!/usr/bin/env python3
"""
Alpha stress test: train TOPO_FNO for 5 epochs with alpha_topo in {0, 10, 100}, same seed.
After each run, evaluate 100 boundary samples and print mean topo_error, mean topo_accuracy.
If these do not change with alpha, topo_loss is not influencing training.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism_ad.dataset import get_dataloaders, data_dir_for_dataset, load_meta
from prism_ad.models import TopoFNO
from prism_ad.training import train
from prism_ad.persistence_eval import wasserstein_d0_d1, _l2_structure_proxy


def evaluate_boundary_topo(run_dir, data_dir, dataset_name, n_samples=100):
    """Load checkpoint from run_dir, compute mean topo_error and topo_accuracy on boundary set."""
    from prism_ad.dataset import load_dataset, load_splits
    data, splits = load_dataset(data_dir)
    meta = load_meta(data_dir)
    boundary_idx = splits.get("boundary_idx", [])
    if not boundary_idx:
        boundary_idx = splits.get("val_idx", [])[:n_samples]
    boundary_idx = boundary_idx[:n_samples]
    params = data["params"]
    grid_truth = data["grid_truth"]
    param_dim = meta["param_dim"]
    grid_size = meta.get("grid_size", 128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from evaluate_models import get_model
    model = get_model(run_dir, param_dim, grid_size, device)
    if model is None:
        return np.nan, np.nan
    topo_errors = []
    with torch.no_grad():
        for i in boundary_idx:
            pred = model(params[i : i + 1].to(device))
            if isinstance(pred, tuple):
                pred = pred[0]
            p = pred.cpu().numpy().squeeze()
            t = grid_truth[i].numpy().squeeze()
            _, _, w = wasserstein_d0_d1(p, t)
            if not np.isfinite(w):
                w = _l2_structure_proxy(p, t)
            topo_errors.append(float(w))
    mean_topo = float(np.mean(topo_errors))
    # d0 from FNO boundary (load FNO run for this dataset)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fno_dir = os.path.join(root, "runs", dataset_name, "fno")
    d0 = 1.0
    if os.path.isfile(os.path.join(fno_dir, "checkpoint.pt")):
        fno_errors = []
        model_fno = get_model(fno_dir, param_dim, grid_size, device)
        with torch.no_grad():
            for i in boundary_idx:
                pred = model_fno(params[i : i + 1].to(device))
                if isinstance(pred, tuple):
                    pred = pred[0]
                p = pred.cpu().numpy().squeeze()
                t = grid_truth[i].numpy().squeeze()
                _, _, w = wasserstein_d0_d1(p, t)
                if np.isfinite(w):
                    fno_errors.append(float(w))
        if fno_errors:
            d0 = float(np.median(fno_errors))
    if d0 <= 0:
        d0 = 1.0
    topo_accuracy = 100 * max(0.0, 1.0 - mean_topo / (2.0 * d0))
    return mean_topo, topo_accuracy


def main():
    dataset_name = "benchmark_operator"
    data_dir = data_dir_for_dataset(dataset_name)
    if not os.path.isfile(os.path.join(data_dir, "meta.json")):
        print("Run gen_data_dataset first for benchmark_operator")
        return
    meta = load_meta(data_dir)
    param_dim = meta["param_dim"]
    grid_size = meta.get("grid_size", 128)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    base_cfg = {
        "model": "topo_fno",
        "run_dir": None,
        "data_dir": data_dir,
        "epochs": 5,
        "batch_size": 32,
        "lr": 1e-3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "mixed_precision": False,
        "fno_modes": 16,
        "fno_layers": 4,
        "fno_channels": 16,
        "lambda_boundary": 2.0,
        "lambda_aux": 0.5,
        "lambda_gamma": 0.2,
        "boundary_delta": 0.05,
        "beta_topo": 1.0,
        "topo_grid": 32,
        "boundary_epsilon": meta.get("boundary_epsilon", 0.05),
    }
    train_loader, val_loader, bound_loader, data, splits = get_dataloaders(data_dir, batch_size=32)
    dataset = (train_loader, val_loader, bound_loader, data, splits)

    results = []
    for alpha in [0, 10, 100]:
        run_dir = os.path.join(root, "runs", dataset_name, f"topo_fno_alpha{alpha}")
        os.makedirs(run_dir, exist_ok=True)
        cfg = {**base_cfg, "run_dir": run_dir, "alpha_topo": float(alpha)}
        print(f"\n=== alpha_topo = {alpha} (5 epochs, same seed={seed}) ===")
        model = TopoFNO(param_dim, grid_size, 16, 16, 4)
        train(model, dataset, cfg)
        # Evaluate 100 boundary samples
        mean_topo, mean_acc = evaluate_boundary_topo(run_dir, data_dir, dataset_name, n_samples=100)
        results.append((alpha, mean_topo, mean_acc))
        print(f"  -> 100 boundary samples: mean topo_error = {mean_topo:.6g}  mean topo_accuracy = {mean_acc:.1f}%")

    print("\n--- Alpha stress test summary ---")
    print("  alpha_topo | mean boundary topo_error | mean boundary topo_accuracy (%)")
    for alpha, mean_topo, mean_acc in results:
        print(f"  {alpha:10d} | {mean_topo:24.6g} | {mean_acc:28.1f}")
    if len(results) >= 2:
        errs = [r[1] for r in results]
        if max(errs) - min(errs) < 1e-6:
            print("  WARNING: boundary topo_error did not change with alpha. Topology loss may not be connected.")
        else:
            print("  boundary topo_error varies with alpha -> topology loss is affecting training.")


if __name__ == "__main__":
    main()
