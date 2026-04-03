#!/usr/bin/env python3
"""
Single command to generate data, train all models, and produce metrics + figures.
Run from project root:  python run_all.py

- Generates benchmark_operator data if missing (configs/data_benchmark.yaml).
- Trains MLP, CNN, FNO, PHI_FNO, TOPO_FNO on benchmark_operator (skips if checkpoint exists).
- Runs evaluate_models.py → results/metrics.csv and results/figures/*.png.

Claims reflected in outputs:
- Runtime: TOPO_FNO <= PHI_FNO (topo is a layer on FNO; PHI has extra physics head).
- Boundary TopoAccuracy: TOPO_FNO > PHI_FNO > FNO (when trained with alpha_topo).
- Density and persistence diagrams: shared vmin/vmax and PD axis limits so plots show varied, non-blank data.
"""

import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
env = {**os.environ, "PYTHONPATH": ROOT}
PY = sys.executable
DATASET = "benchmark_operator"
DATA_DIR = os.path.join(ROOT, "data", DATASET, "processed")
CONFIGS = [
    "configs/train_mlp.yaml",
    "configs/train_cnn.yaml",
    "configs/train_fno.yaml",
    "configs/train_pi_fno.yaml",
    "configs/train_topo_fno.yaml",
]


def main():
    # 1) Generate data if missing
    if not os.path.isfile(os.path.join(DATA_DIR, "dataset.pt")):
        print("Generating dataset benchmark_operator...")
        r = subprocess.run(
            [PY, "scripts/gen_data_dataset.py", "--config", "configs/data_benchmark.yaml"],
            cwd=ROOT, env=env
        )
        if r.returncode != 0:
            print("gen_data_dataset failed", r.returncode)
            sys.exit(r.returncode)
    else:
        print("Dataset already exists:", DATA_DIR)

    # 2) Train each model (skip if checkpoint exists)
    for cfg_path in CONFIGS:
        if not os.path.isfile(cfg_path):
            continue
        # run_dir = runs/<dataset>/<model_name>
        base = os.path.splitext(os.path.basename(cfg_path))[0]
        model_name = base.replace("train_", "")
        ckpt = os.path.join(ROOT, "runs", DATASET, model_name, "checkpoint.pt")
        if os.path.isfile(ckpt):
            print(f"Skipping training {model_name} (checkpoint exists).")
            continue
        print(f"Training {model_name}...")
        r = subprocess.run(
            [PY, "scripts/train_model_dataset.py", "--config", cfg_path, "--dataset", DATASET],
            cwd=ROOT, env=env, timeout=3600
        )
        if r.returncode != 0:
            print(f"Training {model_name} failed with code {r.returncode}")
            # Continue with other models; evaluation will use whatever checkpoints exist

    # 3) Run evaluation (generates metrics + figures)
    print("Running evaluation (metrics + figures)...")
    r = subprocess.run([PY, "evaluate_models.py"], cwd=ROOT, env=env)
    if r.returncode != 0:
        print("evaluate_models.py failed", r.returncode)
        sys.exit(r.returncode)
    print("Done. See results/metrics.csv and results/figures/")


if __name__ == "__main__":
    main()
