# Print verification checklist: speedup, metric table, deltas, figure paths; save results_summary.json.

import json
import os
import glob


def main():
    eval_path = "runs/eval_all_metrics.json"
    runtime_path = "runs/runtime.json"
    all_m = {}
    if os.path.isfile(eval_path):
        with open(eval_path) as f:
            all_m = json.load(f)
    rt = {}
    if os.path.isfile(runtime_path):
        with open(runtime_path) as f:
            rt = json.load(f)

    solver_ms = rt.get("solver_ms_per_sample") or 1.0
    names = ["mlp", "cnn", "fno", "pi_fno", "topo_fno"]
    if not all_m:
        print("No eval metrics found. Run: python scripts/eval_all.py --runs runs/*")
    print("=== SPEEDUP ===")
    for n in names:
        t = rt.get(f"{n}_ms")
        if t is not None:
            su = solver_ms / t
            print(f"  {n}: solver={solver_ms:.2f} ms, model={t:.2f} ms, speedup={su:.1f}x")
    fno_ms = rt.get("fno_ms") or 1e-6
    topo_ms = rt.get("topo_fno_ms") or 1e-6
    print(f"  speedup_FNO = {solver_ms / fno_ms:.1f}")
    print(f"  speedup_TopoFNO = {solver_ms / topo_ms:.1f}")

    print("\n=== ACCURACY (Val / Boundary / OOD MSE) ===")
    for n in names:
        m = all_m.get(n, {})
        print(f"  {n}: Val MSE={m.get('val_mse', 0):.2e}, Boundary MSE={m.get('boundary_mse', 0):.2e}, OOD MSE={m.get('ood_mse', 0):.2e}")

    print("\n=== TOPOLOGY (W0, W1, TopoDist when GUDHI+Persim; else Structure proxy distance) ===")
    for n in names:
        m = all_m.get(n, {})
        w0 = m.get("topo_w0")
        w1 = m.get("topo_w1")
        td = m.get("topo_dist")
        sp = m.get("structure_proxy_distance")
        n0 = m.get("num_H0_features")
        n1 = m.get("num_H1_features")
        print(f"  {n}: W0={w0}, W1={w1}, TopoDist={td}, structure_proxy_distance={sp}, num_H0_features={n0}, num_H1_features={n1}")

    def _topo_val(m):
        return m.get("topo_dist") if m.get("topo_dist") is not None else m.get("structure_proxy_distance") or 0

    print("\n=== IMPROVEMENT DELTAS ===")
    fno_mse = all_m.get("fno", {}).get("val_mse")
    fno_topo = _topo_val(all_m.get("fno", {}))
    pi_mse = all_m.get("pi_fno", {}).get("val_mse")
    pi_topo = _topo_val(all_m.get("pi_fno", {}))
    topo_mse = all_m.get("topo_fno", {}).get("val_mse")
    topo_topo = _topo_val(all_m.get("topo_fno", {}))
    if fno_mse is not None and pi_mse is not None:
        print(f"  PI-FNO vs FNO: Delta MSE = {pi_mse - fno_mse:.2e}, Delta TopoDist = {pi_topo - fno_topo:.2e}")
        if fno_mse:
            print(f"    % MSE = {(pi_mse - fno_mse) / fno_mse * 100:.1f}%")
    if pi_mse is not None and topo_mse is not None:
        print(f"  Topo-FNO vs PI-FNO: Delta MSE = {topo_mse - pi_mse:.2e}, Delta TopoDist = {topo_topo - pi_topo:.2e}")
        if pi_mse:
            print(f"    % MSE = {(topo_mse - pi_mse) / pi_mse * 100:.1f}%")
    if fno_mse is not None and topo_mse is not None:
        print(f"  Topo-FNO vs FNO: Delta MSE = {topo_mse - fno_mse:.2e}, Delta TopoDist = {topo_topo - fno_topo:.2e}")
        if fno_mse:
            print(f"    % MSE = {(topo_mse - fno_mse) / fno_mse * 100:.1f}%")

    fig_paths = []
    for p in ["runs/figs/density_mapping.png", "runs/figs/eigenvalue_cloud_to_density.png", "runs/figs/model_comparison_grid.png", "runs/figs/mse_lies_panel.png", "runs/figs/ph_distance_barchart.png", "runs/figs/boundary_stress.png", "runs/figs/ood_plot.png", "runs/figs/runtime_speedup.png", "runs/figs/improvement_bars.png"]:
        if os.path.isfile(p):
            fig_paths.append(p)
    for d in glob.glob("runs/*/"):
        for f in glob.glob(os.path.join(d, "figs", "*.png")):
            fig_paths.append(f)
    print("\n=== FIGURE PATHS ===")
    for p in sorted(fig_paths):
        print(f"  {p}")

    summary = {
        "speedup": {"solver_ms": solver_ms, "fno_ms": fno_ms, "topo_fno_ms": topo_ms, "speedup_fno": solver_ms / fno_ms, "speedup_topo_fno": solver_ms / topo_ms},
        "val_mse": {n: all_m.get(n, {}).get("val_mse") for n in names},
        "boundary_mse": {n: all_m.get(n, {}).get("boundary_mse") for n in names},
        "ood_mse": {n: all_m.get(n, {}).get("ood_mse") for n in names},
        "topo_dist": {n: all_m.get(n, {}).get("topo_dist") for n in names},
        "structure_proxy_distance": {n: all_m.get(n, {}).get("structure_proxy_distance") for n in names},
        "deltas": {
            "pi_fno_vs_fno": {"delta_mse": (pi_mse - fno_mse) if fno_mse is not None else None, "delta_topo": (pi_topo - fno_topo) if fno_topo is not None else None},
            "topo_fno_vs_pi_fno": {"delta_mse": (topo_mse - pi_mse) if pi_mse is not None else None, "delta_topo": (topo_topo - pi_topo) if pi_topo is not None else None},
            "topo_fno_vs_fno": {"delta_mse": (topo_mse - fno_mse) if fno_mse is not None else None, "delta_topo": (topo_topo - fno_topo) if fno_topo is not None else None},
        },
        "figure_paths": fig_paths,
    }
    os.makedirs("runs", exist_ok=True)
    with open("runs/results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved runs/results_summary.json")


if __name__ == "__main__":
    main()
