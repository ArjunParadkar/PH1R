# Measure solver time per sample and model inference time; save runtime.json and runtime_speedup.png.

import argparse
import json
import os
import time
import numpy as np
import torch

from prism_ad.solver import solve_one
from prism_ad.dataset import load_meta, load_dataset
from prism_ad.models import MLP, CNN, FNO2d, PIFNO, TopoFNO


def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def get_model(name, param_dim, grid_size, device):
    if name == "mlp":
        m = MLP(param_dim, grid_size)
    elif name == "cnn":
        m = CNN(param_dim, grid_size)
    elif name == "fno":
        m = FNO2d(param_dim, grid_size, 16, 16, 4)
    elif name == "pi_fno":
        m = PIFNO(param_dim, grid_size, 16, 16, 4)
    elif name == "topo_fno":
        m = TopoFNO(param_dim, grid_size, 16, 16, 4)
    else:
        return None
    run_dir = os.path.join("runs", name)
    ckpt = os.path.join(run_dir, "checkpoint.pt")
    if os.path.isfile(ckpt):
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        m.load_state_dict(state.get("model", state))
    return m.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/runtime.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    data_cfg = load_config(cfg["data"])
    n_warmup = cfg.get("n_warmup", 5)
    n_samples = cfg.get("n_samples", 50)
    batch_sizes = cfg.get("batch_sizes", [1, 32])
    meta = load_meta("data/processed")
    data, _ = load_dataset("data/processed")
    params = data["params"]
    param_dim = meta["param_dim"]
    grid_size = meta["grid_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_eigs = data_cfg.get("n_eigs", 20)
    tol = data_cfg.get("solver_tol", 1e-8)

    solver_times = []
    for i in range(n_warmup):
        solve_one(params[i].numpy(), n_eigs=n_eigs, tol=tol)
    for i in range(n_samples):
        t0 = time.perf_counter()
        solve_one(params[i].numpy(), n_eigs=n_eigs, tol=tol)
        solver_times.append((time.perf_counter() - t0) * 1000)
    solver_ms = np.mean(solver_times)
    solver_std = np.std(solver_times)
    results = {"solver_ms_per_sample": solver_ms, "solver_std_ms": solver_std}

    model_names = ["mlp", "cnn", "fno", "pi_fno", "topo_fno"]
    for name in model_names:
        if not os.path.isfile(os.path.join("runs", name, "checkpoint.pt")):
            continue
        model = get_model(name, param_dim, grid_size, device)
        if model is None:
            continue
        for bs in batch_sizes:
            if bs > params.shape[0]:
                continue
            p_b = params[:bs].to(device)
            for _ in range(n_warmup):
                with torch.no_grad():
                    _ = model(p_b) if not hasattr(model, "stab_head") else model(p_b, return_stab=False)
            times = []
            for _ in range(n_samples):
                torch.cuda.synchronize() if device.type == "cuda" else None
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = model(p_b) if not hasattr(model, "stab_head") else model(p_b, return_stab=False)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
            key = f"{name}_bs{bs}_ms"
            results[key] = np.mean(times)
            results[key.replace("_ms", "_std")] = np.std(times)
        p1 = params[:1].to(device)
        for _ in range(n_warmup):
            with torch.no_grad():
                _ = model(p1) if not hasattr(model, "stab_head") else model(p1, return_stab=False)
        times1 = []
        for _ in range(n_samples):
            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(p1) if not hasattr(model, "stab_head") else model(p1, return_stab=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times1.append((time.perf_counter() - t0) * 1000)
        results[f"{name}_ms"] = np.mean(times1)
        results[f"{name}_std_ms"] = np.std(times1)

    os.makedirs("runs", exist_ok=True)
    with open("runs/runtime.json", "w") as f:
        json.dump(results, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["mlp", "cnn", "fno", "pi_fno", "topo_fno"]
    model_ms = [results.get(f"{n}_ms", 0) for n in names]
    speedups = [solver_ms / (t or 1e-6) for t in model_ms]
    plt.figure(figsize=(6, 4))
    plt.barh(names, speedups)
    plt.xscale("log")
    plt.xlabel("Speedup (solver / model)")
    plt.title("Runtime speedup")
    plt.tight_layout()
    os.makedirs("runs/figs", exist_ok=True)
    plt.savefig("runs/figs/runtime_speedup.png")
    plt.close()
    print("bench_runtime done", "runs/runtime.json")


if __name__ == "__main__":
    main()
