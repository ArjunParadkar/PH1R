# Train one model on one dataset; run_dir = runs/<dataset_name>/<model_name>.

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import torch

from prism_ad.models import MLP, CNN, FNO2d, PIFNO, TopoFNO
from prism_ad.dataset import get_dataloaders, data_dir_for_dataset, load_meta
from prism_ad.training import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="e.g. configs/train_mlp.yaml")
    p.add_argument("--dataset", required=True, help="benchmark_operator | vlasov_maxwell_gkeyll | fusion_real_lhd")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_dir = data_dir_for_dataset(args.dataset)
    if not os.path.isfile(os.path.join(data_dir, "meta.json")):
        raise FileNotFoundError(f"Run gen_data_dataset first for {args.dataset}")
    meta = load_meta(data_dir)
    param_dim = meta["param_dim"]
    grid_size = meta.get("grid_size", 128)
    model_name = cfg["model"]
    run_dir = os.path.join("runs", args.dataset, model_name)
    cfg["run_dir"] = run_dir
    cfg["data_dir"] = data_dir
    cfg["boundary_epsilon"] = meta.get("boundary_epsilon", 0.05)
    if "splits" not in cfg:
        from prism_ad.dataset import load_splits
        splits = load_splits(data_dir)
        cfg["boundary_epsilon"] = splits.get("boundary_epsilon", 0.05)
    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    if model_name == "mlp":
        model = MLP(param_dim, grid_size)
    elif model_name == "cnn":
        model = CNN(param_dim, grid_size)
    elif model_name == "fno":
        model = FNO2d(param_dim, grid_size, channels=cfg.get("fno_channels", 16), modes=cfg.get("fno_modes", 16), n_layers=cfg.get("fno_layers", 4))
    elif model_name == "pi_fno":
        model = PIFNO(param_dim, grid_size, channels=cfg.get("fno_channels", 16), modes=cfg.get("fno_modes", 16), n_layers=cfg.get("fno_layers", 4))
    elif model_name == "topo_fno":
        model = TopoFNO(param_dim, grid_size, channels=cfg.get("fno_channels", 16), modes=cfg.get("fno_modes", 16), n_layers=cfg.get("fno_layers", 4))
    else:
        raise ValueError(model_name)
    train_loader, val_loader, bound_loader, data, splits = get_dataloaders(data_dir, batch_size=cfg.get("batch_size", 32))
    dataset = (train_loader, val_loader, bound_loader, data, splits)
    train(model, dataset, cfg)
    print("train done", run_dir)

    # After training completes, run evaluation so results/metrics.csv and results/figures/ are updated
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import subprocess
    r = subprocess.run(
        [sys.executable, os.path.join(root, "evaluate_models.py")],
        cwd=root,
        env={**os.environ, "PYTHONPATH": root},
    )
    if r.returncode != 0:
        print("Warning: evaluate_models.py exited with code", r.returncode)


if __name__ == "__main__":
    main()
