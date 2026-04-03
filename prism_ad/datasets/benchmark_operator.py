# Mathematical benchmark: regime-controlled motifs (2-cluster, 3-cluster, ring, merging) with gamma_max crossing 0.

import csv
import json
import os
import numpy as np
import torch

from prism_ad.grid_utils import eigs_to_density

REGIME_A = 0
REGIME_B = 1
REGIME_C = 2
REGIME_D = 3


def _two_clusters(p, n_eigs, omega_min, omega_max, gamma_min, gamma_max):
    o1, o2 = -0.2 + 0.1 * p[0], 0.2 + 0.1 * p[1]
    g_shift = p[2] * 0.15
    eigs = np.concatenate([
        np.linspace(o1 - 0.05, o1 + 0.05, n_eigs // 2) + 1j * (np.linspace(gamma_min, gamma_max, n_eigs // 2) * 0.5 + g_shift),
        np.linspace(o2 - 0.05, o2 + 0.05, n_eigs - n_eigs // 2) + 1j * (np.linspace(gamma_min, gamma_max, n_eigs - n_eigs // 2) * 0.5 + g_shift),
    ])
    return eigs.astype(np.complex128)


def _three_clusters(p, n_eigs, omega_min, omega_max, gamma_min, gamma_max):
    g_shift = p[0] * 0.2
    n = n_eigs // 3
    eigs = np.concatenate([
        -0.25 + 0.05 * np.random.randn(n) + 1j * (np.linspace(-0.1, 0.1, n) + g_shift),
        0.0 + 0.05 * np.random.randn(n) + 1j * (np.linspace(-0.1, 0.15, n) + g_shift),
        0.25 + 0.05 * np.random.randn(n_eigs - 2 * n) + 1j * (np.linspace(-0.1, 0.1, n_eigs - 2 * n) + g_shift),
    ])
    return eigs.astype(np.complex128)


def _ring_annulus(p, n_eigs, omega_min, omega_max, gamma_min, gamma_max):
    g_shift = p[0] * 0.25
    theta = np.linspace(0, 2 * np.pi, n_eigs, endpoint=False)
    r = 0.2 + 0.05 * p[1]
    omega = r * np.cos(theta)
    gamma = r * np.sin(theta) * 0.5 + g_shift
    return (omega + 1j * gamma).astype(np.complex128)


def _merging_clusters(p, n_eigs, omega_min, omega_max, gamma_min, gamma_max):
    t = np.clip((p[0] + 1) / 2, 0.0, 1.0)
    g_shift = p[1] * 0.2
    n1 = int(n_eigs * (1 - t))
    n1 = min(max(n1, 0), n_eigs)
    n2 = n_eigs - n1
    o1, o2 = -0.15 - 0.1 * t, 0.15 + 0.1 * t
    eigs = np.concatenate([
        o1 + 0.02 * np.random.randn(n1) + 1j * (np.linspace(-0.1, 0.1, n1) + g_shift),
        o2 + 0.02 * np.random.randn(n2) + 1j * (np.linspace(-0.1, 0.1, n2) + g_shift),
    ])
    return eigs.astype(np.complex128)


def generate_benchmark_operator(config_path, out_base="data"):
    with open(config_path) as f:
        import yaml
        cfg = yaml.safe_load(f)
    np.random.seed(cfg["seed"])
    dataset_name = cfg["dataset_name"]
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
    regime_weights = cfg.get("regime_weights", [0.25] * 4)
    n_eigs = 24

    out_dir = os.path.join(out_base, dataset_name, "processed")
    os.makedirs(out_dir, exist_ok=True)
    results_dir = os.path.join("runs", "results", dataset_name)
    os.makedirs(results_dir, exist_ok=True)

    regime_generators = [_two_clusters, _three_clusters, _ring_annulus, _merging_clusters]
    n_train = int(n_total * train_frac)
    params = np.random.uniform(param_low, param_high, (n_total, param_dim)).astype(np.float32)
    regimes = np.random.choice(4, size=n_total, p=regime_weights)
    eigs_list = []
    gamma_max_list = []
    grid_truth = []
    omega_stats_list = []

    for i in range(n_total):
        p = params[i]
        r = regimes[i]
        eigs = regime_generators[r](p, n_eigs, omega_min, omega_max, gamma_min, gamma_max)
        eigs_list.append(eigs)
        gm = float(np.max(eigs.imag))
        gamma_max_list.append(gm)
        grid = eigs_to_density(eigs, grid_size, omega_min, omega_max, gamma_min, gamma_max, sigma)
        grid_truth.append(grid)
        omega_stats_list.append({"mean": float(eigs.real.mean()), "std": float(eigs.real.std())})

    gamma_max_arr = np.array(gamma_max_list, dtype=np.float32)
    grid_truth = np.stack(grid_truth, axis=0)[:, np.newaxis, :, :]

    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n_total))
    boundary_mask = np.abs(gamma_max_arr) < boundary_epsilon
    boundary_idx = np.where(boundary_mask)[0].tolist()
    ood_params = (params * ood_shift_factor).astype(np.float32)
    grid_ood = []
    gamma_max_ood = []
    for i in range(min(n_total, len(ood_params))):
        p = ood_params[i]
        r = regimes[i]
        eigs = regime_generators[r](p, n_eigs, omega_min, omega_max, gamma_min, gamma_max)
        gm = float(np.max(eigs.imag))
        gamma_max_ood.append(gm)
        grid_ood.append(eigs_to_density(eigs, grid_size, omega_min, omega_max, gamma_min, gamma_max, sigma))
    grid_ood = np.stack(grid_ood, axis=0)[:, np.newaxis, :, :]
    gamma_max_ood = np.array(gamma_max_ood, dtype=np.float32)

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
        "dataset_name": dataset_name,
        "has_gamma_max": True,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    splits = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "boundary_idx": boundary_idx,
        "boundary_epsilon": boundary_epsilon,
        "ood_shift_factor": ood_shift_factor,
    }
    with open(os.path.join(out_dir, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)

    torch.save({
        "params": torch.from_numpy(params),
        "eigs": eigs_list,
        "grid_truth": torch.from_numpy(grid_truth),
        "gamma_max": torch.from_numpy(gamma_max_arr),
        "regime_id": torch.from_numpy(regimes),
        "ood_params": torch.from_numpy(ood_params),
        "grid_ood": torch.from_numpy(grid_ood),
        "gamma_max_ood": torch.from_numpy(gamma_max_ood),
    }, os.path.join(out_dir, "dataset.pt"))

    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["idx", "dataset", "regime_id", "gamma_max", "n_eigs"] + [f"p{i}" for i in range(param_dim)]
        header += ["omega_mean", "omega_std"]
        w.writerow(header)
        for i in range(n_total):
            row = [i, dataset_name, int(regimes[i]), gamma_max_list[i], n_eigs] + params[i].tolist()
            row += [omega_stats_list[i]["mean"], omega_stats_list[i]["std"]]
            w.writerow(row)

    regime_counts = [int((regimes == k).sum()) for k in range(4)]
    boundary_count = len(boundary_idx)
    hist, _ = np.histogram(gamma_max_arr, bins=10, range=(-0.3, 0.3))
    summary_path = os.path.join(results_dir, "dataset_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime_id", "count"])
        for k in range(4):
            w.writerow([k, regime_counts[k]])
        w.writerow(["boundary_count", boundary_count])
        w.writerow(["gamma_max_hist_bins", ",".join(map(str, hist))])

    return out_dir
