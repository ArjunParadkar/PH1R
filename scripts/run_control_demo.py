#!/usr/bin/env python3
"""
PRISM-AD Control Integration Demo
===================================
Demonstrates that TopoFNO can be embedded in a real-time feedback control
loop for a tokamak, adjusting toroidal field (TF), poloidal field (PF),
central solenoid (CS), and error-field-correction (EFC) coil currents to
drive an unstable plasma back to stability.

Three controllers are compared:
  1. No control      -- plasma evolves freely (instability grows)
  2. Proportional    -- heuristic gain on TF current only (no model)
  3. TopoFNO-MPC     -- gradient-based MPC using TopoFNO as forward model

Figures produced (results/figures/control/):
  ctrl_01_gamma_trajectory.png   -- gamma_max vs time for all three controllers
  ctrl_02_coil_adjustments.png   -- per-coil delta_u time series (MPC only)
  ctrl_03_spectrum_evolution.png -- density grids at t=0,T/4,T/2,T (MPC)
  ctrl_04_phase_portrait.png     -- trajectory in (beta_N, q95) space
  ctrl_05_mpc_cost_surface.png   -- 2-D cost landscape in (I_TF, I_CS) space
  ctrl_06_inference_latency.png  -- TopoFNO inference time vs solver time
  ctrl_07_topology_vs_time.png   -- number of spectral components over time
  ctrl_08_multi_scenario.png     -- MPC on 6 different initial conditions

Usage:
    cd /home/god/Desktop/Arenda_Projects/PRSM-AD
    python scripts/run_control_demo.py [--dataset d3d_real] [--n_steps 60] [--seed 7]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism_ad.models import MLP, CNN, FNO2d, PIFNO, TopoFNO
from prism_ad.dataset import data_dir_for_dataset, load_dataset, load_meta
from prism_ad.control.coil_model import LinearizedCoilModel, COIL_NAMES, PARAM_NAMES
from prism_ad.control.gamma_extractor import DifferentiableGammaMax
from prism_ad.control.mpc_controller import GradientMPCController, BaselineProportionalController

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OUT_DIR = "results/figures/control"


def _load_topo_fno(dataset_name: str, device: torch.device):
    run_dir = os.path.join("runs", dataset_name, "topo_fno")
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"No TopoFNO checkpoint for {dataset_name}. "
            f"Run python run_full_evaluation.py first."
        )
    data_dir = data_dir_for_dataset(dataset_name)
    meta = load_meta(data_dir)
    param_dim = meta["param_dim"]
    grid_size = meta.get("grid_size", 128)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = state.get("model", state)
    channels = sd["embed.weight"].shape[0]
    n_layers = sum(1 for k in sd if k.startswith("blocks.") and k.endswith(".pw.bias"))
    modes = sd["blocks.0.fourier.weights"].shape[2]
    is_v2 = "spectral_attn.net.0.weight" in sd

    if is_v2:
        model = TopoFNO(param_dim, grid_size, channels=channels, modes=modes, n_layers=n_layers)
    else:
        model = PIFNO(param_dim, grid_size, channels=channels, modes=modes, n_layers=n_layers)
    model.load_state_dict(sd)
    return model.to(device).eval(), meta


def _gamma_from_density(rho_np: np.ndarray, gamma_min: float, gamma_max: float) -> float:
    """Soft-argmax gamma_max from a [1,H,W] numpy density array."""
    r = rho_np.squeeze()          # [H, W]
    W = r.shape[1]
    gamma_vals = np.linspace(gamma_min, gamma_max, W)
    marginal = r.sum(axis=0)      # [W]  marginal over omega
    tau = 0.05
    w = np.exp(marginal / tau)
    w /= w.sum()
    return float((w * gamma_vals).sum())


def _count_spectral_components(rho_np: np.ndarray, threshold_frac: float = 0.15) -> int:
    """Estimate number of spectral components (H0 Betti number proxy)."""
    r = rho_np.squeeze()
    peak = r.max()
    binary = (r > threshold_frac * peak).astype(np.uint8)
    # Simple connected-component count via flood fill proxy:
    # Count transitions in the marginal
    marginal = r.sum(axis=0)
    marginal /= marginal.max() + 1e-12
    in_peak = marginal > 0.1
    transitions = int(np.sum(np.diff(in_peak.astype(int)) > 0))
    return max(1, transitions)


def _run_episode(
    model,
    coil_model: LinearizedCoilModel,
    extractor: DifferentiableGammaMax,
    controller,
    p_init: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
    device: torch.device,
    meta: dict,
    add_noise: bool = True,
    mode: str = "mpc",  # "mpc" | "proportional" | "none"
) -> dict:
    """
    Run one closed-loop episode.

    Returns dict with keys:
      gamma_traj   [n_steps]
      p_traj       [n_steps, 8]
      du_traj      [n_steps, 8]
      rho_snapshots  list of [1,1,H,W] numpy arrays at t=0,T/4,T/2,T
      n_components [n_steps]
      inference_ms [n_steps]
    """
    if hasattr(controller, "reset"):
        controller.reset()
    if hasattr(coil_model, "reset"):
        coil_model.reset()

    gamma_traj, p_traj, du_traj, comp_traj, inf_traj = [], [], [], [], []
    snap_steps = [0, n_steps // 4, n_steps // 2, n_steps - 1]
    rho_snapshots = []

    p = p_init.copy()
    gamma_min = meta.get("gamma_min", -0.3)
    gamma_max_val = meta.get("gamma_max", 0.3)

    for t in range(n_steps):
        p_tensor = torch.tensor(p, dtype=torch.float32, device=device).unsqueeze(0)

        t0 = time.perf_counter()
        with torch.no_grad():
            rho = model(p_tensor)
            if isinstance(rho, tuple):
                rho = rho[0]
        inf_ms = (time.perf_counter() - t0) * 1000

        rho_np = rho.cpu().numpy()
        gmax = _gamma_from_density(rho_np, gamma_min, gamma_max_val)

        gamma_traj.append(gmax)
        p_traj.append(p.copy())
        inf_traj.append(inf_ms)
        comp_traj.append(_count_spectral_components(rho_np))

        if t in snap_steps:
            rho_snapshots.append(rho_np.copy())

        # Controller decision
        if mode == "none":
            delta_u = np.zeros(8, dtype=np.float32)
        elif mode == "proportional":
            delta_u = controller.step(gmax)
        else:  # mpc
            delta_u, _ = controller.step(p)

        du_traj.append(delta_u.copy())

        # Plant step
        p, _ = coil_model.apply(p, delta_u, rng=rng, add_noise=add_noise)

    return {
        "gamma_traj": np.array(gamma_traj),
        "p_traj": np.array(p_traj),
        "du_traj": np.array(du_traj),
        "rho_snapshots": rho_snapshots,
        "n_components": np.array(comp_traj),
        "inference_ms": np.array(inf_traj),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

COLORS = {
    "none": "#CC3311",
    "proportional": "#EE7733",
    "mpc": "#009988",
}
LABELS = {
    "none": "No control",
    "proportional": "Proportional (TF only)",
    "mpc": "TopoFNO-MPC (all coils)",
}


def fig_gamma_trajectory(episodes: dict, n_steps: int, dt_ms: float = 2.0):
    fig, ax = plt.subplots(figsize=(8, 4))
    t_ms = np.arange(n_steps) * dt_ms
    for mode, ep in episodes.items():
        ax.plot(t_ms, ep["gamma_traj"], color=COLORS[mode], label=LABELS[mode], lw=2)
    ax.axhline(0, color="k", lw=1, ls="--", label="Stability boundary (γ=0)")
    ax.axhline(0.02, color="k", lw=0.8, ls=":", alpha=0.5, label="Control threshold")
    ax.fill_between(t_ms, 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.3,
                    alpha=0.07, color="red")
    ax.set_xlabel("Time (ms)", fontsize=11)
    ax.set_ylabel("γ_max  (growth rate)", fontsize=11)
    ax.set_title("Closed-loop stabilisation: γ_max trajectory", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, t_ms[-1])
    fig.tight_layout()
    return fig


def fig_coil_adjustments(ep: dict, n_steps: int, dt_ms: float = 2.0):
    du = ep["du_traj"]   # [T, 8]
    t_ms = np.arange(n_steps) * dt_ms
    fig, axes = plt.subplots(4, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()
    coil_colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for i, ax in enumerate(axes):
        ax.plot(t_ms, du[:, i], color=coil_colors[i], lw=1.5)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_ylabel(f"Δ{COIL_NAMES[i]}", fontsize=9)
        ax.set_ylim(-0.18, 0.18)
        ax.tick_params(labelsize=8)
    for ax in axes[-2:]:
        ax.set_xlabel("Time (ms)", fontsize=9)
    fig.suptitle("TopoFNO-MPC: coil current adjustments Δu(t)", fontsize=12)
    fig.tight_layout()
    return fig


def fig_spectrum_evolution(ep: dict, meta: dict):
    snaps = ep["rho_snapshots"]   # list of [1,1,H,W]
    n = len(snaps)
    omega_min = meta.get("omega_min", -0.5)
    omega_max_v = meta.get("omega_max", 0.5)
    gamma_min = meta.get("gamma_min", -0.3)
    gamma_max_v = meta.get("gamma_max", 0.3)

    titles = [f"t = 0 ms", f"t = T/4", f"t = T/2", f"t = T  (stabilised)"]
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
    for i, (ax, snap) in enumerate(zip(axes, snaps)):
        img = snap.squeeze()
        im = ax.imshow(
            img,
            origin="lower",
            aspect="auto",
            extent=[gamma_min, gamma_max_v, omega_min, omega_max_v],
            cmap="inferno",
        )
        ax.axvline(0, color="w", lw=0.8, ls="--", alpha=0.7)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xlabel("γ (growth rate)", fontsize=8)
        if i == 0:
            ax.set_ylabel("ω (freq)", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Eigenmode spectral density: TopoFNO-MPC evolution", fontsize=12)
    fig.tight_layout()
    return fig


def fig_phase_portrait(episodes: dict):
    fig, ax = plt.subplots(figsize=(6, 5))
    for mode, ep in episodes.items():
        traj = ep["p_traj"]   # [T, 8]
        ax.plot(traj[:, 0], traj[:, 1], color=COLORS[mode], label=LABELS[mode],
                lw=1.5, alpha=0.85)
        ax.scatter(traj[0, 0], traj[0, 1], color=COLORS[mode], marker="o", s=80, zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=COLORS[mode], marker="*", s=120, zorder=5)
    # Stability boundary (approximate): beta_N_norm ~ 0 corresponds to beta_N ~ beta_N_crit
    ax.axvline(0.0, color="k", lw=1.2, ls="--", alpha=0.6, label="Approx. stability boundary")
    ax.fill_betweenx([-1.5, 1.5], 0, 1.5, alpha=0.07, color="red", label="Unstable region")
    ax.set_xlabel("beta_N (normalised)", fontsize=11)
    ax.set_ylabel("q95 (normalised)", fontsize=11)
    ax.set_title("Phase portrait in (β_N, q₉₅) space", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    fig.tight_layout()
    return fig


def fig_mpc_cost_surface(model, extractor, p_base: np.ndarray, meta: dict,
                          device: torch.device):
    """2-D cost landscape sliced in the (I_TF, I_CS) plane."""
    from prism_ad.control.coil_model import _J_NOMINAL
    J = torch.tensor(_J_NOMINAL, dtype=torch.float32, device=device)
    alpha = 0.4
    n = 30
    tf_vals = np.linspace(-0.5, 0.5, n)
    cs_vals = np.linspace(-0.5, 0.5, n)
    cost_grid = np.zeros((n, n), dtype=np.float32)

    p_t = torch.tensor(p_base, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        for i, tf in enumerate(tf_vals):
            for j, cs in enumerate(cs_vals):
                du = torch.zeros(8, device=device)
                du[0] = tf
                du[1] = cs
                delta_p = alpha * (J @ du)
                p_new = (p_t.squeeze(0) + delta_p).clamp(-1.2, 1.2).unsqueeze(0)
                rho = model(p_new)
                if isinstance(rho, tuple):
                    rho = rho[0]
                cost_grid[i, j] = extractor(rho).item()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.contourf(cs_vals, tf_vals, cost_grid, levels=20, cmap="RdYlGn_r")
    ax.contour(cs_vals, tf_vals, cost_grid, levels=[0.0], colors="k",
               linewidths=2, linestyles="--")
    plt.colorbar(im, ax=ax, label="Predicted γ_max")
    ax.set_xlabel("ΔI_CS (central solenoid)", fontsize=11)
    ax.set_ylabel("ΔI_TF (toroidal field)", fontsize=11)
    ax.set_title("MPC cost landscape: γ_max(β_N, q₉₅) via TF/CS coils\n"
                 "(dashed = stability boundary γ=0)", fontsize=10)
    ax.scatter([0], [0], color="w", marker="x", s=120, zorder=5, label="Current setpoint")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def fig_inference_latency(ep_mpc: dict, solver_ms: float = 50.0):
    inf_ms = ep_mpc["inference_ms"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(inf_ms, color="#009988", lw=1.5, label=f"TopoFNO (mean {inf_ms.mean():.2f} ms)")
    ax.axhline(solver_ms, color="#CC3311", lw=2, ls="--",
               label=f"Eigenvalue solver ({solver_ms:.0f} ms/sample)")
    ax.axhline(2.0, color="k", lw=1, ls=":", alpha=0.6, label="Typical control-loop budget (2 ms)")
    ax.set_xlabel("Control step", fontsize=11)
    ax.set_ylabel("Inference time (ms)", fontsize=11)
    ax.set_title("TopoFNO inference latency vs solver", fontsize=12)
    ax.legend(fontsize=9)
    speedup = solver_ms / inf_ms.mean()
    ax.text(0.98, 0.85, f"Speedup: {speedup:.0f}×",
            transform=ax.transAxes, ha="right", fontsize=12,
            color="#009988", fontweight="bold")
    fig.tight_layout()
    return fig


def fig_topology_vs_time(episodes: dict, n_steps: int, dt_ms: float = 2.0):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    t_ms = np.arange(n_steps) * dt_ms
    for mode, ep in episodes.items():
        ax.step(t_ms, ep["n_components"], color=COLORS[mode],
                label=LABELS[mode], lw=1.8, where="post")
    ax.set_xlabel("Time (ms)", fontsize=11)
    ax.set_ylabel("Spectral components (β₀ proxy)", fontsize=11)
    ax.set_title("Number of spectral peaks (topological Betti-0 proxy)", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, t_ms[-1])
    ax.set_yticks([1, 2, 3])
    fig.tight_layout()
    return fig


def fig_multi_scenario(model, coil_model_cls, extractor, J_tensor, meta: dict,
                        device: torch.device, rng: np.random.Generator,
                        n_steps: int = 60):
    """Run MPC on 6 different initial conditions spanning the instability space."""
    scenarios = [
        {"p0": np.array([0.55, -0.3, 0.1, 0.0, 0.2, 0.1, 0.0, 0.05], np.float32),
         "label": "Mild NTM onset"},
        {"p0": np.array([0.80, -0.5, 0.2, 0.1, 0.3, 0.2, 0.0, 0.0], np.float32),
         "label": "Strong NTM"},
        {"p0": np.array([0.60, -0.2, 0.0, 0.3, 0.5, 0.4, 0.0, 0.1], np.float32),
         "label": "High coupling"},
        {"p0": np.array([0.50, -0.4, 0.3, -0.1, 0.1, 0.0, 0.0, -0.1], np.float32),
         "label": "Low-q unstable"},
        {"p0": np.array([0.70, -0.1, -0.1, 0.2, 0.2, 0.3, 0.1, 0.0], np.float32),
         "label": "TAE + NTM"},
        {"p0": np.array([0.45, -0.6, 0.1, 0.0, 0.1, 0.1, 0.0, 0.15], np.float32),
         "label": "Near-boundary"},
    ]

    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(scenarios)))
    dt_ms = 2.0
    t_ms = np.arange(n_steps) * dt_ms

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, (sc, ax) in enumerate(zip(scenarios, axes)):
        cm = coil_model_cls()
        ctrl = GradientMPCController(
            model=model, coil_jacobian=J_tensor, gamma_extractor=extractor,
            threshold_activate=0.02, alpha=0.4, lambda_u=0.05, lambda_s=0.02,
            n_opt_steps=30, lr_opt=0.05, rate_limit=0.15, device=device,
        )
        ep = _run_episode(model, cm, extractor, ctrl, sc["p0"], n_steps,
                          rng, device, meta, add_noise=True, mode="mpc")
        ax.plot(t_ms, ep["gamma_traj"], color=cmap[idx], lw=2)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.fill_between(t_ms, 0, 0.4, alpha=0.06, color="red")
        ax.set_title(sc["label"], fontsize=9)
        ax.set_ylim(-0.25, 0.4)
        ax.tick_params(labelsize=7)
        final = ep["gamma_traj"][-1]
        stabilised = final < 0.0
        status = "✓ stable" if stabilised else "✗ unstable"
        ax.text(0.97, 0.93, status, transform=ax.transAxes, ha="right",
                fontsize=8, color="green" if stabilised else "red")

    for ax in axes[3:]:
        ax.set_xlabel("Time (ms)", fontsize=8)
    for ax in [axes[0], axes[3]]:
        ax.set_ylabel("γ_max", fontsize=9)

    fig.suptitle("TopoFNO-MPC: stabilisation across 6 initial conditions", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="d3d_real",
                        choices=["d3d_real", "fusion_real_tcv", "fusion_real_lhd"])
    parser.add_argument("--n_steps", type=int, default=60,
                        help="Number of control steps (each step ~ 2 ms)")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Dataset: {args.dataset}  |  Steps: {args.n_steps}")

    # Load model
    print("Loading TopoFNO checkpoint...")
    model, meta = _load_topo_fno(args.dataset, device)
    gamma_min = meta.get("gamma_min", -0.3)
    gamma_max_v = meta.get("gamma_max", 0.3)

    extractor = DifferentiableGammaMax(
        gamma_min=gamma_min, gamma_max=gamma_max_v,
        grid_size=meta.get("grid_size", 128), tau=0.05,
    ).to(device)

    # Coil model + Jacobian
    coil_model = LinearizedCoilModel(noise_std=0.008)
    J_tensor = coil_model.jacobian_tensor(device)

    # Controllers
    mpc_ctrl = GradientMPCController(
        model=model, coil_jacobian=J_tensor, gamma_extractor=extractor,
        threshold_activate=0.02, alpha=0.4,
        lambda_u=0.05, lambda_s=0.02,
        n_opt_steps=40, lr_opt=0.05, rate_limit=0.15, device=device,
    )
    prop_ctrl = BaselineProportionalController(gain=0.5, threshold=0.02)

    # Initial condition: clearly unstable (beta_N above critical)
    p_init = np.array([0.65, -0.35, 0.15, 0.05, 0.25, 0.15, 0.0, 0.08], np.float32)
    print(f"Initial plasma params: {dict(zip(PARAM_NAMES, p_init.tolist()))}")

    # Run all three episodes from the same initial condition
    episodes = {}
    for mode, ctrl in [("none", None), ("proportional", prop_ctrl), ("mpc", mpc_ctrl)]:
        print(f"Running episode: {mode}...")
        cm = LinearizedCoilModel(noise_std=0.008)
        ep = _run_episode(
            model, cm, extractor, ctrl, p_init, args.n_steps,
            rng=np.random.default_rng(args.seed + 10),
            device=device, meta=meta, add_noise=True, mode=mode,
        )
        episodes[mode] = ep
        g_final = ep["gamma_traj"][-1]
        stabilised = g_final < 0.0
        print(f"  gamma_max: {ep['gamma_traj'][0]:.3f} -> {g_final:.3f}  "
              f"({'STABLE' if stabilised else 'UNSTABLE'})")

    # Measure solver time for latency comparison
    from prism_ad.solver import solve_one
    data_dir = data_dir_for_dataset(args.dataset)
    data, _ = load_dataset(data_dir)
    p_bench = data["params"][0].numpy()
    solver_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        solve_one(p_bench, n_eigs=20, tol=1e-8)
        solver_times.append((time.perf_counter() - t0) * 1000)
    solver_ms = float(np.mean(solver_times))
    print(f"Solver: {solver_ms:.1f} ms | TopoFNO: {episodes['mpc']['inference_ms'].mean():.2f} ms "
          f"| Speedup: {solver_ms / episodes['mpc']['inference_ms'].mean():.0f}x")

    # ---- Figure 1: gamma trajectory ----
    print("Generating figures...")
    fig = fig_gamma_trajectory(episodes, args.n_steps)
    fig.savefig(f"{OUT_DIR}/ctrl_01_gamma_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: coil adjustments ----
    fig = fig_coil_adjustments(episodes["mpc"], args.n_steps)
    fig.savefig(f"{OUT_DIR}/ctrl_02_coil_adjustments.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 3: spectrum evolution ----
    fig = fig_spectrum_evolution(episodes["mpc"], meta)
    fig.savefig(f"{OUT_DIR}/ctrl_03_spectrum_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 4: phase portrait ----
    fig = fig_phase_portrait(episodes)
    fig.savefig(f"{OUT_DIR}/ctrl_04_phase_portrait.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 5: cost surface ----
    fig = fig_mpc_cost_surface(model, extractor, p_init, meta, device)
    fig.savefig(f"{OUT_DIR}/ctrl_05_mpc_cost_surface.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 6: inference latency ----
    fig = fig_inference_latency(episodes["mpc"], solver_ms=solver_ms)
    fig.savefig(f"{OUT_DIR}/ctrl_06_inference_latency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 7: topology vs time ----
    fig = fig_topology_vs_time(episodes, args.n_steps)
    fig.savefig(f"{OUT_DIR}/ctrl_07_topology_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 8: multi-scenario ----
    fig = fig_multi_scenario(
        model, LinearizedCoilModel, extractor, J_tensor, meta, device,
        rng=np.random.default_rng(args.seed + 99), n_steps=args.n_steps,
    )
    fig.savefig(f"{OUT_DIR}/ctrl_08_multi_scenario.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Summary JSON ----
    summary = {
        "dataset": args.dataset,
        "n_steps": args.n_steps,
        "dt_ms": 2.0,
        "solver_ms": solver_ms,
        "topo_fno_inference_ms_mean": float(episodes["mpc"]["inference_ms"].mean()),
        "speedup": float(solver_ms / episodes["mpc"]["inference_ms"].mean()),
        "gamma_init": float(episodes["mpc"]["gamma_traj"][0]),
        "gamma_final": {m: float(ep["gamma_traj"][-1]) for m, ep in episodes.items()},
        "stabilised": {m: bool(ep["gamma_traj"][-1] < 0.0) for m, ep in episodes.items()},
        "figures": [
            f"{OUT_DIR}/ctrl_0{i+1}_*.png" for i in range(8)
        ],
    }
    with open(f"{OUT_DIR}/control_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nControl demo complete.")
    print(f"  Speedup over solver : {summary['speedup']:.0f}x")
    print(f"  Final gamma_max     : "
          + "  |  ".join(f"{m}: {v:.3f}" for m, v in summary["gamma_final"].items()))
    print(f"  Stabilised          : "
          + "  |  ".join(f"{m}: {v}" for m, v in summary["stabilised"].items()))
    print(f"  Figures -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
