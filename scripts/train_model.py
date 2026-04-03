# Train one model from config: MLP, CNN, FNO, PI-FNO, Topo-FNO.

import argparse
import os
import yaml
import torch

from prism_ad.models import MLP, CNN, FNO2d, PIFNO, TopoFNO
from prism_ad.dataset import get_dataloaders, load_meta
from prism_ad.training import train


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    data_cfg = load_config(cfg["data"]) if isinstance(cfg["data"], str) else cfg.get("data_cfg", {})
    meta = load_meta("data/processed")
    param_dim = meta["param_dim"]
    grid_size = meta["grid_size"]
    model_name = cfg["model"]
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
    train_loader, val_loader, bound_loader, data, splits = get_dataloaders("data/processed", batch_size=cfg.get("batch_size", 32))
    dataset = (train_loader, val_loader, bound_loader, data, splits)
    cfg["boundary_epsilon"] = splits.get("boundary_epsilon", 0.05)
    artifacts = train(model, dataset, cfg)
    print("train done", artifacts["run_dir"])


if __name__ == "__main__":
    main()
