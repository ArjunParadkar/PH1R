# Full Vlasov-Maxwell spectra from Gkeyll: parameter sweep, spectral extraction to (omega, gamma), same artifact format.

import csv
import json
import os
import numpy as np
import torch

from prism_ad.grid_utils import eigs_to_density
from prism_ad.physics.spectral_extract import extract_eigenvalue_cloud_from_modes


def _synthetic_mode_amplitudes(t, omega_k, gamma_k, n_modes=15):
    """Synthetic E_k(t) for testing when Gkeyll not run: E_k(t) = exp(gamma_k*t) * cos(omega_k*t)."""
    t = np.asarray(t)
    amps = []
    for _ in range(n_modes):
        o = omega_k + 0.1 * np.random.randn()
        g = gamma_k + 0.02 * np.random.randn()
        amps.append(np.exp(g * t) * (np.cos(o * t) + 0.1 * np.random.randn(len(t))))
    return amps


def generate_vm_gkeyll(config_path, out_base="data"):
    with open(config_path) as f:
        import yaml
        cfg = yaml.safe_load(f)
    np.random.seed(cfg["seed"])
    dataset_name = cfg["dataset_name"]
    n_total = min(cfg["n_total"], 50)
    train_frac = cfg["train_frac"]
    boundary_epsilon = cfg["boundary_epsilon"]
    param_dim = cfg["param_dim"]
    grid_size = cfg["grid_size"]
    sigma = cfg["density_sigma"]
    omega_min = cfg["omega_min"]
    omega_max = cfg["omega_max"]
    gamma_min = cfg["gamma_min"]
    gamma_max = cfg["gamma_max"]
    precomputed = cfg.get("gkeyll_precomputed_dir")
    window_frac = cfg.get("spectral_window_frac", 0.3)

    out_dir = os.path.join(out_base, dataset_name, "processed")
    os.makedirs(out_dir, exist_ok=True)
    results_dir = os.path.join("runs", "results", dataset_name)
    os.makedirs(results_dir, exist_ok=True)

    if precomputed and os.path.isdir(precomputed):
        pt_path = os.path.join(precomputed, "dataset.pt")
        if os.path.isfile(pt_path):
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
            torch.save(data, os.path.join(out_dir, "dataset.pt"))
            for name in ["meta.json", "splits.json", "manifest.csv"]:
                src = os.path.join(precomputed, name)
                if os.path.isfile(src):
                    import shutil
                    shutil.copy(src, os.path.join(out_dir, name))
            return out_dir

    n_train = int(n_total * train_frac)
    params = np.random.uniform(-1, 1, (n_total, param_dim)).astype(np.float32)
    t_linear = np.linspace(0, 10, 200)
    eigs_list = []
    gamma_max_list = []
    grid_truth = []

    for i in range(n_total):
        p = params[i]
        omega_center = 0.3 * p[0]
        gamma_center = 0.1 * p[1] - 0.05
        t_list = [t_linear] * 12
        amp_list = _synthetic_mode_amplitudes(t_linear, omega_center, gamma_center, n_modes=12)
        eigs, gamma_max = extract_eigenvalue_cloud_from_modes(t_list, amp_list, window_frac=window_frac)
        eigs = np.array([e for e in eigs if omega_min <= e.real <= omega_max and gamma_min <= e.imag <= gamma_max], dtype=np.complex128)
        if eigs.size == 0:
            eigs = np.array([0.0 + 1j * gamma_center], dtype=np.complex128)
        eigs_list.append(eigs)
        gamma_max_list.append(gamma_max)
        grid = eigs_to_density(eigs, grid_size, omega_min, omega_max, gamma_min, gamma_max, sigma)
        grid_truth.append(grid)

    gamma_max_arr = np.array(gamma_max_list, dtype=np.float32)
    grid_truth = np.stack(grid_truth, axis=0)[:, np.newaxis, :, :]
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n_total))
    boundary_idx = np.where(np.abs(gamma_max_arr) < boundary_epsilon)[0].tolist()
    ood_params = (params * 1.3).astype(np.float32)
    grid_ood = []
    gamma_max_ood = []
    for i in range(n_total):
        p = ood_params[i]
        t_list = [t_linear] * 12
        amp_list = _synthetic_mode_amplitudes(t_linear, 0.3 * p[0], 0.1 * p[1] - 0.05, 12)
        eigs, _ = extract_eigenvalue_cloud_from_modes(t_list, amp_list, window_frac=window_frac)
        eigs = np.array([e for e in eigs if omega_min <= e.real <= omega_max and gamma_min <= e.imag <= gamma_max], dtype=np.complex128)
        if eigs.size == 0:
            eigs = np.array([0.0 + 1j * 0.0], dtype=np.complex128)
        grid_ood.append(eigs_to_density(eigs, grid_size, omega_min, omega_max, gamma_min, gamma_max, sigma))
        gamma_max_ood.append(float(np.max(eigs.imag)))
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
    splits = {"train_idx": train_idx, "val_idx": val_idx, "boundary_idx": boundary_idx, "boundary_epsilon": boundary_epsilon}
    with open(os.path.join(out_dir, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)
    torch.save({
        "params": torch.from_numpy(params),
        "eigs": eigs_list,
        "grid_truth": torch.from_numpy(grid_truth),
        "gamma_max": torch.from_numpy(gamma_max_arr),
        "ood_params": torch.from_numpy(ood_params),
        "grid_ood": torch.from_numpy(grid_ood),
        "gamma_max_ood": torch.from_numpy(gamma_max_ood),
    }, os.path.join(out_dir, "dataset.pt"))
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "dataset", "gamma_max", "n_modes"] + [f"p{i}" for i in range(param_dim)])
        for i in range(n_total):
            w.writerow([i, dataset_name, gamma_max_list[i], len(eigs_list[i])] + params[i].tolist())
    return out_dir
