# Generate dataset: sample params, run baseline eigen-solver, build 128x128 density grids, save dataset.pt, meta.json, splits.json.

import argparse
import json
import os
import time
import numpy as np
import torch

from prism_ad.solver import solve_one
from prism_ad.grid_utils import eigs_to_density


def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    np.random.seed(cfg["seed"])
    n_total = cfg["n_total"]
    train_frac = cfg["train_frac"]
    boundary_epsilon = cfg["boundary_epsilon"]
    ood_shift_factor = cfg["ood_shift_factor"]
    param_dim = cfg["param_dim"]
    param_low = cfg["param_low"]
    param_high = cfg["param_high"]
    grid_size = cfg["grid_size"]
    sigma = cfg["density_sigma"]
    omega_min = cfg["omega_min"]
    omega_max = cfg["omega_max"]
    gamma_min = cfg["gamma_min"]
    gamma_max = cfg["gamma_max"]
    n_eigs = cfg["n_eigs"]
    tol = cfg["solver_tol"]

    n_train = int(n_total * train_frac)
    params = np.random.uniform(param_low, param_high, (n_total, param_dim)).astype(np.float32)
    ood_params = params * ood_shift_factor

    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "grid_size": grid_size,
        "omega_min": omega_min,
        "omega_max": omega_max,
        "gamma_min": gamma_min,
        "gamma_max": gamma_max,
        "density_sigma": sigma,
        "density_sigma_units": "physical (omega, gamma)",
        "normalization": "sum-to-1",
        "param_dim": param_dim,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    eigs_list = []
    gamma_max_list = []
    grid_truth = []
    t0 = time.perf_counter()
    for i in range(n_total):
        lam = solve_one(params[i], n_eigs=n_eigs, tol=tol)
        eigs_list.append(lam)
        gm = float(np.max(lam.imag))
        gamma_max_list.append(gm)
        grid = eigs_to_density(lam, grid_size, omega_min, omega_max, gamma_min, gamma_max, sigma)
        grid_truth.append(grid)
    t1 = time.perf_counter()
    gamma_max_arr = np.array(gamma_max_list, dtype=np.float32)
    grid_truth = np.stack(grid_truth, axis=0)[:, np.newaxis, :, :]

    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n_total)
    boundary_mask = np.abs(gamma_max_arr) < boundary_epsilon
    boundary_idx = np.where(boundary_mask)[0].tolist()
    ood_params = ood_params.astype(np.float32)

    eigs_ood = []
    gamma_max_ood = []
    grid_ood = []
    for i in range(min(n_total, len(ood_params))):
        lam = solve_one(ood_params[i], n_eigs=n_eigs, tol=tol)
        eigs_ood.append(lam)
        gamma_max_ood.append(float(np.max(lam.imag)))
        grid_ood.append(eigs_to_density(lam, grid_size, omega_min, omega_max, gamma_min, gamma_max, sigma))
    gamma_max_ood = np.array(gamma_max_ood, dtype=np.float32)
    grid_ood = np.stack(grid_ood, axis=0)[:, np.newaxis, :, :]

    splits = {
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "boundary_idx": boundary_idx,
        "boundary_epsilon": boundary_epsilon,
        "ood_shift_factor": ood_shift_factor,
    }

    torch.save({
        "params": torch.from_numpy(params),
        "eigs": eigs_list,
        "grid_truth": torch.from_numpy(grid_truth),
        "gamma_max": torch.from_numpy(gamma_max_arr),
        "ood_params": torch.from_numpy(ood_params),
        "grid_ood": torch.from_numpy(grid_ood),
        "gamma_max_ood": torch.from_numpy(gamma_max_ood),
    }, os.path.join(out_dir, "dataset.pt"))
    with open(os.path.join(out_dir, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)

    print(f"gen_data done. solver total time {t1-t0:.2f}s samples={n_total}")


if __name__ == "__main__":
    main()
