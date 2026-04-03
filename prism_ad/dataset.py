# Load dataset.pt and splits; support data/<dataset_name>/processed and optional transition_score.

import json
import os
import torch


def data_dir_for_dataset(dataset_name, base="data"):
    return os.path.join(base, dataset_name, "processed")


def load_meta(data_dir):
    with open(os.path.join(data_dir, "meta.json")) as f:
        return json.load(f)


def load_splits(data_dir):
    with open(os.path.join(data_dir, "splits.json")) as f:
        return json.load(f)


def load_dataset(data_dir):
    # dataset.pt contains tensors and numpy arrays from our generators; safe to load without weights_only
    data = torch.load(os.path.join(data_dir, "dataset.pt"), map_location="cpu", weights_only=False)
    splits = load_splits(data_dir)
    return data, splits


class PRISMDataset(torch.utils.data.Dataset):
    def __init__(self, params, grid_truth, gamma_max, indices, transition_score=None):
        self.params = params
        self.grid_truth = grid_truth
        self.gamma_max = gamma_max
        self.indices = indices
        self.transition_score = transition_score

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        if self.transition_score is not None:
            return self.params[j], self.grid_truth[j], self.gamma_max[j], j, self.transition_score[j]
        return self.params[j], self.grid_truth[j], self.gamma_max[j], j


def get_dataloaders(data_dir, batch_size=32):
    data, splits = load_dataset(data_dir)
    data["meta"] = load_meta(data_dir)
    params = data["params"]
    grid_truth = data["grid_truth"]
    gamma_max = data["gamma_max"]
    transition_score = data.get("transition_score")
    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    boundary_idx = splits.get("boundary_idx") or []
    train_ds = PRISMDataset(params, grid_truth, gamma_max, train_idx, transition_score)
    val_ds = PRISMDataset(params, grid_truth, gamma_max, val_idx, transition_score)
    bound_ds = PRISMDataset(params, grid_truth, gamma_max, boundary_idx, transition_score) if boundary_idx else val_ds
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    bound_loader = torch.utils.data.DataLoader(bound_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, bound_loader, data, splits
