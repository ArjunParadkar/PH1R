# TCV Grey2 tokamak magnetic fluctuation dataset.
# Models TCV double-tearing mode physics: two distinct MHD modes on separate rational
# surfaces (q=1 and q=2). Topology change at boundary: 1 spectral component -> 2.
# gamma_max from beta_pol - beta_pol_crit (Rutherford-like criterion with shape effects).
# "Grey2" refers to a TCV plasma shape configuration emphasising double tearing.

import csv
import json
import os
import numpy as np
import torch

from prism_ad.physics.stft_utils import stft_power, spectrogram_to_density_128x128


def _colored_noise(n_pts, rng, alpha=1.5):
    """1/f^alpha noise via FFT amplitude shaping."""
    white = rng.standard_normal(n_pts)
    spectrum = np.fft.rfft(white)
    n_freq = len(spectrum)
    freqs = np.arange(1, n_freq + 1, dtype=np.float64)
    amplitude = freqs ** (-alpha / 2.0)
    spectrum[1:] *= amplitude[:-1]
    spectrum[0] = 0.0
    colored = np.fft.irfft(spectrum, n=n_pts)
    s = np.std(colored)
    return colored / (s + 1e-12)


def _tcv_timeseries(p, fs, n_pts, rng):
    """
    Generate a TCV Grey2-like magnetic fluctuation signal.

    Physics encoded (TCV: R=0.88m, a=0.25m, B~1.54T):
    - kappa   (p[0]): elongation [1.2, 2.0]. Higher kappa stabilises kink, shapes modes.
    - delta   (p[1]): triangularity [0.1, 0.6]. Affects rational surface positions.
    - q_edge  (p[2]): edge safety factor [3.0, 5.5]. Controls rational surface locations.
    - beta_pol(p[3]): poloidal beta [0.20, 0.85]. Primary drive for tearing modes.
    - m1_ctrl (p[4]): primary mode frequency controller.
    - m2_ctrl (p[5]): secondary mode frequency controller.
    - coupling(p[6]): mode-mode coupling coefficient [0, 1].
    - j_peak  (p[7]): current profile peaking. Enhances double-tearing drive.

    Double-tearing onset: beta_pol > beta_pol_crit AND peaked current (j_peak > 0.3)
    gamma_max = (beta_pol - beta_pol_crit)*scale + j_peak*j_scale

    KEY TOPOLOGICAL FEATURE:
    - Stable:    sawtooth only (1 spectral peak around f_saw)
    - Boundary:  primary tearing mode emerging (1 peak at f_tear1; topology change if f_tear1 != f_saw)
    - Unstable:  TWO distinct peaks at f_tear1 and f_tear2 (two connected components!)

    This 1->2 component transition at the stability boundary is exactly what
    topology-aware training is designed to preserve.

    Returns (signal [n_pts], gamma_max float).
    """
    # Physical parameters
    kappa   = 1.2 + 0.4  * (p[0] + 1.0) / 2.0    # [1.2, 1.6]
    delta   = 0.1 + 0.25 * (p[1] + 1.0) / 2.0    # [0.1, 0.35]
    q_edge  = 3.5 + 1.0  * (p[2] + 1.0) / 2.0    # [3.0, 4.5]
    beta_pol = 0.20 + 0.325 * (p[3] + 1.0)        # [0.20, 0.85]
    m1_ctrl  = float(p[4])                          # [-1, 1]
    m2_ctrl  = float(p[5])                          # [-1, 1]
    coup     = (p[6] + 1.0) / 2.0                  # [0, 1]
    j_peak   = (p[7] + 1.0) / 2.0                  # [0, 1] current peaking

    # Simplified Rutherford onset criterion with shape correction
    # Double tearing requires both high beta_pol AND peaked current
    beta_pol_crit = 0.525  # mid-range -> natural boundary
    j_crit = 0.35
    gamma_raw = ((beta_pol - beta_pol_crit) * 0.7
                 + max(0.0, j_peak - j_crit) * 0.3
                 - kappa * 0.02  # elongation weakly stabilising
                 + delta * 0.03)  # triangularity weakly destabilising
    gamma_raw = float(np.clip(gamma_raw, -0.3, 0.3))

    t = np.arange(n_pts) / fs

    # --- Ohmic background turbulence ---
    x = _colored_noise(n_pts, rng, alpha=1.5) * 0.04

    # --- Sawtooth (q=1 surface, m/n = 1/1 relaxation oscillation) ---
    # Frequency increases with lower density / lower q in core
    f_saw_kHz = 1.5 + 1.5 * kappa + 0.5 * (1.0 - q_edge / 4.5)
    f_saw = f_saw_kHz * 1e3    # 2.5 – 4.5 kHz range
    saw_amp = 0.06 + 0.02 * (1.0 - beta_pol / 0.85)
    x += saw_amp * np.sin(2 * np.pi * f_saw * t)
    # Sawtooth harmonic
    x += saw_amp * 0.25 * np.sin(2 * np.pi * 2 * f_saw * t)

    # --- PRIMARY tearing mode: m/n = 2/1 on q=2 surface ---
    # Frequency: mode rotation in lab frame ∝ E×B drift
    f_tear1_kHz = 5.0 + 3.5 * (q_edge / 4.5) + m1_ctrl * 0.8  # 5 – 10 kHz
    f_tear1 = f_tear1_kHz * 1e3

    # --- SECONDARY tearing mode: m/n = 3/1 or 1/1 depending on q profile ---
    # This is what makes TCV "double tearing" - two separate resonant surfaces
    # f_tear2 is notably different from f_tear1 (not a harmonic)
    f_tear2_kHz = f_tear1_kHz * 0.45 + 0.8 + m2_ctrl * 0.5    # ~2.3 – 5 kHz (distinct from f_tear1)
    f_tear2 = f_tear2_kHz * 1e3

    boundary_margin = -0.06
    if gamma_raw > boundary_margin:
        # PRIMARY mode grows first (lower threshold)
        tear1_amp = max(0.0, gamma_raw - boundary_margin) * 2.5
        phi1 = rng.uniform(0, 2 * np.pi)
        # Amplitude has time-dependent envelope: builds up across the window
        envelope1 = np.exp(np.clip(gamma_raw * t * 30, 0.0, 2.0))
        envelope1 = np.minimum(envelope1, np.exp(2.0))
        x += tear1_amp * envelope1 * np.cos(2 * np.pi * f_tear1 * t + phi1)

        # SECONDARY mode: slightly higher threshold (double-tearing requires more drive)
        secondary_threshold = 0.03
        if gamma_raw > secondary_threshold:
            tear2_amp = max(0.0, gamma_raw - secondary_threshold) * 2.0
            phi2 = rng.uniform(0, 2 * np.pi)
            envelope2 = np.exp(np.clip((gamma_raw - secondary_threshold) * t * 25, 0.0, 2.0))
            envelope2 = np.minimum(envelope2, np.exp(2.0))
            x += tear2_amp * envelope2 * np.cos(2 * np.pi * f_tear2 * t + phi2)

            # Mode coupling: beat frequency (topological signature of two interacting modes)
            if coup > 0.35:
                f_beat = abs(f_tear1 - f_tear2)
                beat_amp = coup * min(tear1_amp, tear2_amp) * 0.20
                x += beat_amp * np.cos(2 * np.pi * f_beat * t + rng.uniform(0, 2 * np.pi))

    # --- TAE-like mode at high frequency ---
    f_TAE_kHz = 20.0 + 15.0 * beta_pol / 0.85    # 20 – 35 kHz
    f_TAE = f_TAE_kHz * 1e3
    x += 0.006 * np.cos(2 * np.pi * f_TAE * t + rng.uniform(0, 2 * np.pi))

    return x.astype(np.float32), gamma_raw


def generate_tcv(config_path, out_base="data"):
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

    fs = float(cfg.get("fs", 50000.0))    # 50 kHz: captures up to 25 kHz (covers all TCV modes)
    n_pts = int(cfg.get("n_pts", 4096))   # 81.9 ms window

    out_dir = os.path.join(out_base, dataset_name, "processed")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join("runs", "results", dataset_name), exist_ok=True)

    params = rng.uniform(-1.0, 1.0, (n_total, param_dim)).astype(np.float32)
    grid_truth_list = []
    gamma_max_list = []

    for i in range(n_total):
        sig, gm = _tcv_timeseries(params[i], fs, n_pts, rng)
        f, t_ax, S = stft_power(sig, fs, nperseg=nperseg)
        grid = spectrogram_to_density_128x128(f, t_ax, S, grid_size=grid_size)
        grid_truth_list.append(grid)
        gamma_max_list.append(gm)

    grid_truth = np.stack(grid_truth_list, axis=0)[:, np.newaxis, :, :]
    gamma_max_arr = np.array(gamma_max_list, dtype=np.float32)

    n_train = int(n_total * train_frac)
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n_total))
    boundary_idx = np.where(np.abs(gamma_max_arr) < boundary_epsilon)[0].tolist()

    # OOD: increase beta_pol (higher drive, more modes)
    ood_params = params.copy()
    ood_params[:, 3] = np.clip(params[:, 3] * ood_shift_factor, -1.0, 1.0)  # scale beta_pol
    grid_ood_list = []
    gamma_max_ood_list = []
    for i in range(n_total):
        sig, gm = _tcv_timeseries(ood_params[i], fs, n_pts, rng)
        f, t_ax, S = stft_power(sig, fs, nperseg=nperseg)
        grid_ood_list.append(spectrogram_to_density_128x128(f, t_ax, S, grid_size=grid_size))
        gamma_max_ood_list.append(gm)
    grid_ood = np.stack(grid_ood_list, axis=0)[:, np.newaxis, :, :]
    gamma_max_ood = np.array(gamma_max_ood_list, dtype=np.float32)

    meta = {
        "dataset_name": dataset_name,
        "grid_size": grid_size,
        "param_dim": param_dim,
        "has_gamma_max": True,
        "normalization": "sum-to-1",
        "gamma_min": -0.3,
        "gamma_max": 0.3,
        "omega_min": 0.0,
        "omega_max": fs / 2.0,
        "fs_hz": fs,
        "n_pts": n_pts,
        "nperseg": nperseg,
        "physics": (
            "TCV Grey2 double-tearing mode. "
            "gamma_max = (beta_pol - beta_pol_crit)*0.7 + j_peak_contribution. "
            "Topology: stable=1 peak (sawtooth), boundary=primary tearing emerging, "
            "unstable=2 distinct peaks (m/n=2/1 + m/n=3/1 tearing surfaces)."
        ),
        "boundary_epsilon": boundary_epsilon,
        "tcv_config": "Grey2 (high-kappa, negative-triangularity-accessible shape)",
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

    print(f"  TCV Grey2: {n_total} samples, {len(boundary_idx)} boundary, "
          f"{(gamma_max_arr > boundary_epsilon).sum()} unstable, "
          f"{(gamma_max_arr < -boundary_epsilon).sum()} stable")
    return out_dir
