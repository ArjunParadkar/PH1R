# DIII-D (D3D) tokamak magnetic fluctuation dataset.
# Models Mirnov coil signals: broadband ITG turbulence + NTM mode onset + sawtooth + TAE.
# gamma_max is derived from beta_N - beta_N_crit (Rutherford-like onset criterion).
# Topological structure: 0 peaks (stable) -> 1 peak (NTM onset, boundary) -> 1-2 peaks + harmonic (unstable).

import csv
import json
import os
import numpy as np
import torch

from prism_ad.physics.stft_utils import stft_power, spectrogram_to_density_128x128


def _colored_noise(n_pts, rng, alpha=2.0):
    """1/f^alpha noise via FFT amplitude shaping. Returns unit-variance signal."""
    white = rng.standard_normal(n_pts)
    spectrum = np.fft.rfft(white)    # length n_pts//2 + 1
    n_freq = len(spectrum)
    freqs = np.arange(1, n_freq + 1, dtype=np.float64)
    amplitude = freqs ** (-alpha / 2.0)
    spectrum[1:] *= amplitude[:-1]   # skip DC, align lengths
    spectrum[0] = 0.0                # zero DC
    colored = np.fft.irfft(spectrum, n=n_pts)
    s = np.std(colored)
    return colored / (s + 1e-12)


def _d3d_timeseries(p, fs, n_pts, rng):
    """
    Generate a DIII-D-like Mirnov coil (magnetics) signal.

    Physics encoded:
    - beta_N (p[0]): normalised beta. NTM onset at beta_N > beta_N_crit (~mid-range).
    - q95   (p[1]): safety factor at 95% flux surface. Controls mode rotation freq.
    - n_e   (p[2]): line-averaged density. Modulates background turbulence level.
    - f_off (p[3]): fine-tunes TAE frequency (energetic particle content proxy).
    - amp   (p[4]): mode amplitude scale factor.
    - coup  (p[5]): NTM harmonic coupling coefficient.
    - broad (p[6]): broadband turbulence amplitude.
    - stoch (p[7]): stochastic offset to gamma_max (finite-size fluctuations).

    gamma_max = (beta_N - beta_N_crit) * scale + p[7] * noise_scale
    Linear model gives gamma_max in [-0.3, 0.3] spanning stable/boundary/unstable.

    Returns (signal [n_pts], gamma_max float).
    """
    # Map p in [-1,1] to physical quantities
    beta_N = 1.25 + 0.75 * p[0]          # [0.5, 2.0]
    q95    = 3.5  + 1.0  * p[1]           # [2.5, 4.5]
    n_e_n  = (p[2] + 1.0) / 2.0           # [0, 1] normalised density
    f_off  = p[3] * 0.2                    # fractional freq offset
    amp_sc = 0.8 + 0.4 * (p[4] + 1) / 2  # [0.8, 1.2]
    coup   = (p[5] + 1.0) / 2.0           # [0, 1]
    broad  = 0.08 + 0.06 * (p[6] + 1) / 2  # [0.08, 0.14]

    # Simplified Rutherford-like onset: gamma ∝ beta_N - beta_crit
    beta_crit = 1.25  # mid-range -> natural boundary at ~50% of parameter space
    gamma_raw = (beta_N - beta_crit) * 0.4 + p[7] * 0.025
    gamma_raw = float(np.clip(gamma_raw, -0.3, 0.3))

    t = np.arange(n_pts) / fs

    # --- Background ITG/TEM turbulence (1/f^2 power spectrum) ---
    x = _colored_noise(n_pts, rng, alpha=2.0) * broad

    # --- Sawtooth (q=1 surface, always present for tokamak) ---
    f_saw = (2.0 + 2.5 * (1.0 - q95 / 5.0)) * 1e3   # 1.5 – 3.5 kHz
    saw_amp = 0.04 + 0.03 * n_e_n
    x += saw_amp * np.sin(2 * np.pi * f_saw * t)

    # --- NTM mode (appears when gamma_raw > -boundary_margin) ---
    f_NTM = (8.0 + 10.0 * (beta_N - 0.5) / 1.5) * 1e3 * (1.0 + f_off)  # 8 – 18 kHz
    ntm_threshold = -0.05  # mode starts to grow slightly before gamma_max > 0
    if gamma_raw > ntm_threshold:
        # Amplitude scales with distance past onset
        base_amp = max(0.0, gamma_raw - ntm_threshold) * amp_sc * 3.0
        # Saturated exponential growth profile over the time window
        growth = np.exp(np.clip(gamma_raw * t * 40, 0.0, 2.5))
        growth = np.minimum(growth, np.exp(2.5))  # saturate
        phi0 = rng.uniform(0, 2 * np.pi)
        x += base_amp * growth * np.cos(2 * np.pi * f_NTM * t + phi0)
        # First harmonic (mode coupling)
        if coup > 0.25:
            harm_amp = coup * base_amp * 0.35
            x += harm_amp * np.cos(2 * np.pi * 2 * f_NTM * t + rng.uniform(0, 2 * np.pi))

    # --- TAE mode at high frequency (energetic particle driven) ---
    # f_TAE ∝ V_A / (2qR): small variation via p[3]
    f_TAE = (35.0 + 25.0 * (4.0 / q95) * (1.0 + f_off * 0.5)) * 1e3  # 30 – 60 kHz range in 100 kHz band
    tae_amp = 0.008 + 0.006 * (p[3] + 1) / 2
    x += tae_amp * np.cos(2 * np.pi * f_TAE * t + rng.uniform(0, 2 * np.pi))

    return x.astype(np.float32), gamma_raw


def generate_d3d(config_path, out_base="data"):
    with open(config_path) as f:
        import yaml
        cfg = yaml.safe_load(f)

    rng = np.random.default_rng(cfg["seed"])
    dataset_name = cfg["dataset_name"]
    n_total = int(cfg.get("n_total", 200))
    train_frac = float(cfg.get("train_frac", 0.8))
    boundary_epsilon = float(cfg.get("boundary_epsilon", 0.04))
    param_dim = int(cfg.get("param_dim", 8))
    grid_size = int(cfg.get("grid_size", 128))
    nperseg = int(cfg.get("stft_nperseg", 256))
    ood_shift_factor = float(cfg.get("ood_shift_factor", 1.2))

    fs = float(cfg.get("fs", 100000.0))   # 100 kHz sampling rate
    n_pts = int(cfg.get("n_pts", 4096))   # 40.96 ms window

    out_dir = os.path.join(out_base, dataset_name, "processed")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join("runs", "results", dataset_name), exist_ok=True)

    params = rng.uniform(-1.0, 1.0, (n_total, param_dim)).astype(np.float32)
    grid_truth_list = []
    gamma_max_list = []

    for i in range(n_total):
        sig, gm = _d3d_timeseries(params[i], fs, n_pts, rng)
        f, t_ax, S = stft_power(sig, fs, nperseg=nperseg)
        grid = spectrogram_to_density_128x128(f, t_ax, S, grid_size=grid_size)
        grid_truth_list.append(grid)
        gamma_max_list.append(gm)

    grid_truth = np.stack(grid_truth_list, axis=0)[:, np.newaxis, :, :]  # [N,1,H,W]
    gamma_max_arr = np.array(gamma_max_list, dtype=np.float32)

    n_train = int(n_total * train_frac)
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n_total))
    boundary_idx = np.where(np.abs(gamma_max_arr) < boundary_epsilon)[0].tolist()

    # OOD: shift beta_N higher (more unstable operating point)
    ood_params = params.copy()
    ood_params[:, 0] = np.clip(params[:, 0] * ood_shift_factor, -1.0, 1.0)  # scale beta_N
    grid_ood_list = []
    gamma_max_ood_list = []
    for i in range(n_total):
        sig, gm = _d3d_timeseries(ood_params[i], fs, n_pts, rng)
        f, t_ax, S = stft_power(sig, fs, nperseg=nperseg)
        grid_ood_list.append(spectrogram_to_density_128x128(f, t_ax, S, grid_size=grid_size))
        gamma_max_ood_list.append(gm)
    grid_ood = np.stack(grid_ood_list, axis=0)[:, np.newaxis, :, :]
    gamma_max_ood = np.array(gamma_max_ood_list, dtype=np.float32)

    # Frequency axis bounds (for meta)
    omega_min_hz = 0.0
    omega_max_hz = fs / 2.0

    meta = {
        "dataset_name": dataset_name,
        "grid_size": grid_size,
        "param_dim": param_dim,
        "has_gamma_max": True,
        "normalization": "sum-to-1",
        "gamma_min": -0.3,
        "gamma_max": 0.3,
        "omega_min": omega_min_hz,
        "omega_max": omega_max_hz,
        "fs_hz": fs,
        "n_pts": n_pts,
        "nperseg": nperseg,
        "physics": "DIII-D NTM onset: gamma_max = (beta_N - beta_N_crit)*0.4 + noise",
        "boundary_epsilon": boundary_epsilon,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f_out:
        json.dump(meta, f_out, indent=2)

    splits = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "boundary_idx": boundary_idx,
        "boundary_epsilon": boundary_epsilon,
        "ood_shift_factor": ood_shift_factor,
    }
    with open(os.path.join(out_dir, "splits.json"), "w") as f_out:
        json.dump(splits, f_out, indent=2)

    torch.save({
        "params": torch.from_numpy(params),
        "grid_truth": torch.from_numpy(grid_truth),
        "gamma_max": torch.from_numpy(gamma_max_arr),
        "ood_params": torch.from_numpy(ood_params),
        "grid_ood": torch.from_numpy(grid_ood),
        "gamma_max_ood": torch.from_numpy(gamma_max_ood),
    }, os.path.join(out_dir, "dataset.pt"))

    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f_out:
        w = csv.writer(f_out)
        w.writerow(["idx", "dataset", "gamma_max", "regime"] + [f"p{i}" for i in range(param_dim)])
        for i in range(n_total):
            gm = gamma_max_list[i]
            regime = "stable" if gm < -boundary_epsilon else ("boundary" if abs(gm) <= boundary_epsilon else "unstable")
            w.writerow([i, dataset_name, gm, regime] + params[i].tolist())

    print(f"  D3D: {n_total} samples, {len(boundary_idx)} boundary, "
          f"{(gamma_max_arr > boundary_epsilon).sum()} unstable, "
          f"{(gamma_max_arr < -boundary_epsilon).sum()} stable")
    return out_dir
