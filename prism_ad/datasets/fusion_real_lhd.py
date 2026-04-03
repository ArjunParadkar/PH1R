# Real fusion LHD (NIFS LHD Open Data): STFT spectrogram -> 128x128 density; p from summary features; boundary from transition score.

import csv
import json
import os
import numpy as np
import torch

from prism_ad.physics.stft_utils import stft_power, spectrogram_to_density_128x128


def _synthetic_timeseries(n_pts, fs, dominant_freq, bandwidth_power, rng):
    """Synthetic diagnostic-like signal when LHD fetch not run."""
    t = np.arange(n_pts) / fs
    x = np.sin(2 * np.pi * dominant_freq * t) + 0.3 * rng.standard_normal(n_pts)
    x += 0.2 * np.sin(2 * np.pi * (dominant_freq * 2) * t)
    return x.astype(np.float32)


def generate_lhd(config_path, out_base="data"):
    with open(config_path) as f:
        import yaml
        cfg = yaml.safe_load(f)
    rng = np.random.default_rng(cfg["seed"])
    dataset_name = cfg["dataset_name"]
    n_total = min(cfg["n_total"], 100)
    train_frac = cfg["train_frac"]
    boundary_epsilon = cfg.get("boundary_epsilon", 0.1)
    param_dim = cfg["param_dim"]
    grid_size = cfg["grid_size"]
    nperseg = cfg.get("stft_nperseg", 256)
    transition_percentile = cfg.get("transition_percentile", 50)

    out_dir = os.path.join(out_base, dataset_name, "processed")
    os.makedirs(out_dir, exist_ok=True)
    results_dir = os.path.join("runs", "results", dataset_name)
    os.makedirs(results_dir, exist_ok=True)

    fs = 1000.0
    n_pts = 2048
    params = rng.uniform(-1, 1, (n_total, param_dim)).astype(np.float32)
    grid_truth = []
    transition_scores = []

    for i in range(n_total):
        p = params[i]
        dominant_freq = 50 + 30 * p[0]
        bandwidth_power = 0.2 + 0.3 * (p[1] + 1) / 2
        x = _synthetic_timeseries(n_pts, fs, dominant_freq, bandwidth_power, rng)
        f, t, S = stft_power(x, fs, nperseg=nperseg)
        grid = spectrogram_to_density_128x128(f, t, S, grid_size=grid_size)
        grid_truth.append(grid)
        transition_scores.append(float(np.median(S) * (1 + p[2])))

    grid_truth = np.stack(grid_truth, axis=0)[:, np.newaxis, :, :]
    transition_scores = np.array(transition_scores, dtype=np.float32)
    gamma_max = np.zeros(n_total, dtype=np.float32)
    thresh = np.percentile(transition_scores, transition_percentile)
    boundary_idx = np.where(np.abs(transition_scores - thresh) < boundary_epsilon * (transition_scores.max() - transition_scores.min() + 1e-6))[0].tolist()
    n_train = int(n_total * train_frac)
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n_total))

    transition_median = float(np.median(transition_scores))
    meta = {
        "grid_size": grid_size,
        "param_dim": param_dim,
        "dataset_name": dataset_name,
        "has_gamma_max": False,
        "has_transition_score": True,
        "transition_median": transition_median,
        "normalization": "sum-to-1",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    splits = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "boundary_idx": boundary_idx,
        "boundary_epsilon": boundary_epsilon,
    }
    with open(os.path.join(out_dir, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)
    ood_params = (params * 1.2).astype(np.float32)
    grid_ood = []
    for i in range(n_total):
        p = ood_params[i]
        x = _synthetic_timeseries(n_pts, fs, 50 + 30 * p[0], 0.2 + 0.3 * (p[1] + 1) / 2, rng)
        f, t, S = stft_power(x, fs, nperseg=nperseg)
        grid_ood.append(spectrogram_to_density_128x128(f, t, S, grid_size=grid_size))
    grid_ood = np.stack(grid_ood, axis=0)[:, np.newaxis, :, :]
    torch.save({
        "params": torch.from_numpy(params),
        "grid_truth": torch.from_numpy(grid_truth),
        "gamma_max": torch.from_numpy(gamma_max),
        "transition_score": torch.from_numpy(transition_scores),
        "ood_params": torch.from_numpy(ood_params),
        "grid_ood": torch.from_numpy(grid_ood),
        "gamma_max_ood": torch.from_numpy(gamma_max),
    }, os.path.join(out_dir, "dataset.pt"))
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "dataset", "transition_score"] + [f"p{i}" for i in range(param_dim)])
        for i in range(n_total):
            w.writerow([i, dataset_name, transition_scores[i]] + params[i].tolist())
    return out_dir
