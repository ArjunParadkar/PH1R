# Unified train(model, dataset, config) -> artifacts: checkpoint, metrics.jsonl, curves, summary.json.

import json
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from prism_ad.topology_loss import TopologyLoss


def gamma_max_to_stab_label(gamma_max, delta):
    stable = (gamma_max < -delta).long()
    unstable = (gamma_max > delta).long()
    boundary = ((gamma_max >= -delta) & (gamma_max <= delta)).long()
    return boundary * 0 + stable * 1 + unstable * 2


def gamma_max_from_density(pred, gamma_min, gamma_max, device):
    """Soft-argmax over gamma axis: mass_per_gamma = pred.sum(dim=2), gamma_pred = (mass * gamma_coords).sum / mass.sum."""
    B = pred.shape[0]
    G = pred.shape[3]
    mass = pred.squeeze(1).sum(dim=1)
    gamma_coords = torch.linspace(gamma_min, gamma_max, G, device=device, dtype=pred.dtype)
    mass = mass + 1e-8
    gamma_pred = (mass * gamma_coords).sum(dim=1) / mass.sum(dim=1)
    return gamma_pred


def train_one_epoch(model, loaders, device, optimizer, scaler, config, topo_fn=None, meta=None, epoch=0):
    model.train()
    train_loader = loaders["train"]
    total_mse = 0.0
    total_mse_boundary = 0.0
    total_aux = 0.0
    total_L_gamma = 0.0
    total_topo = 0.0
    total_loss_sum = 0.0
    n = 0
    lambda_boundary = config.get("lambda_boundary", 0.0)
    lambda_aux = config.get("lambda_aux", 0.0)
    lambda_gamma = config.get("lambda_gamma", 0.0)
    alpha_topo_final = float(config.get("alpha_topo", 0.0))
    warmup_epochs = int(config.get("alpha_topo_warmup", 20))
    if warmup_epochs > 0 and alpha_topo_final > 0:
        alpha_topo = alpha_topo_final * min(1.0, epoch / warmup_epochs)
    else:
        alpha_topo = alpha_topo_final
    boundary_delta = config.get("boundary_delta", 0.05)
    boundary_epsilon = config.get("boundary_epsilon", 0.05)
    has_gamma_max = (meta or {}).get("has_gamma_max", True)
    grad_norm_sq_sum = 0.0
    grad_norm_count = 0

    for batch in train_loader:
        if len(batch) == 5:
            p, truth, gm, _, boundary_val = batch
            boundary_val = boundary_val.to(device)
        else:
            p, truth, gm, _ = batch
            boundary_val = gm
        p = p.to(device)
        truth = truth.to(device)
        gm = gm.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=config.get("mixed_precision", False) and torch.cuda.is_available()):
            if hasattr(model, "stab_head"):
                pred, y_stab = model(p, return_stab=True)
                aux_loss = F.cross_entropy(y_stab, gamma_max_to_stab_label(gm, boundary_delta).clamp(0, 2)) if has_gamma_max else torch.tensor(0.0, device=device)
            else:
                pred = model(p)
                aux_loss = torch.tensor(0.0, device=device)
            mse = F.mse_loss(pred, truth)
            if len(batch) == 5 and (meta or {}).get("transition_median") is not None:
                thresh = (meta or {}).get("transition_median", 0.0)
                b = torch.exp(-torch.abs(boundary_val - thresh) / boundary_epsilon).view(-1, 1, 1, 1)
            else:
                b = torch.exp(-torch.abs(boundary_val) / boundary_epsilon).view(-1, 1, 1, 1)
            b = b.to(pred.device)
            mse_boundary = (b * (pred - truth) ** 2).mean()
            L_gamma = torch.tensor(0.0, device=device)
            if lambda_gamma > 0 and has_gamma_max and meta:
                g_min = meta.get("gamma_min", -0.3)
                g_max = meta.get("gamma_max", 0.3)
                gamma_pred = gamma_max_from_density(pred, g_min, g_max, device)
                L_gamma = F.l1_loss(gamma_pred, gm)
            loss = mse + lambda_boundary * mse_boundary + lambda_aux * aux_loss + lambda_gamma * L_gamma
            topo_loss_val = torch.tensor(0.0, device=device)
            if alpha_topo > 0 and topo_fn is not None:
                topo_loss_val = topo_fn(pred, truth)
                boundary_topo_weight = config.get("boundary_topo_weight", 1.0)
                if boundary_topo_weight != 1.0 and has_gamma_max:
                    frac_boundary = (torch.abs(gm) < boundary_epsilon).float().mean().item()
                    topo_scale = 1.0 + (boundary_topo_weight - 1.0) * frac_boundary
                    topo_loss_val = topo_loss_val * topo_scale
                loss = loss + alpha_topo * topo_loss_val
            loss_val = loss.item()
            # Sanity: computed total must match loss
            computed = mse.item() + lambda_boundary * mse_boundary.item() + lambda_aux * aux_loss.item() + lambda_gamma * L_gamma.item() + alpha_topo * topo_loss_val.item()
            assert abs(computed - loss_val) < max(1e-4, 1e-4 * abs(loss_val)), f"Loss mismatch: computed={computed} loss={loss_val}"
        scaler.scale(loss).backward()
        # Unscale grads before clipping so clip threshold is in true gradient units
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm_sq = 0.0
        for pp in model.parameters():
            if pp.grad is not None:
                grad_norm_sq += pp.grad.detach().data.norm().item() ** 2
        if grad_norm_sq > 0:
            grad_norm_sq_sum += grad_norm_sq
            grad_norm_count += 1
        scaler.step(optimizer)
        scaler.update()
        bs = p.shape[0]
        total_mse += mse.item() * bs
        total_mse_boundary += mse_boundary.item() * bs
        total_aux += aux_loss.item() * bs
        total_L_gamma += L_gamma.item() * bs
        total_topo += topo_loss_val.item() * bs
        total_loss_sum += loss_val * bs
        n += bs
    n = max(n, 1)
    grad_norm = (grad_norm_sq_sum / max(grad_norm_count, 1)) ** 0.5
    L_s = total_mse / n
    L_b = total_mse_boundary / n
    L_a = total_aux / n
    L_g = total_L_gamma / n
    L_t = total_topo / n
    total_mean = total_loss_sum / n
    computed_sum = L_s + lambda_boundary * L_b + lambda_aux * L_a + lambda_gamma * L_g + alpha_topo * L_t
    assert abs(computed_sum - total_mean) < max(1e-3, 1e-3 * abs(total_mean)), f"Epoch loss mismatch: computed={computed_sum} total={total_mean}"
    return {
        "train_mse": L_s,
        "mse_boundary": L_b,
        "aux_loss": L_a,
        "L_gamma": L_g,
        "topo_loss": L_t,
        "total_loss": total_mean,
        "computed_sum": computed_sum,
        "grad_norm": grad_norm,
        "lambda_boundary": lambda_boundary,
        "lambda_aux": lambda_aux,
        "lambda_gamma": lambda_gamma,
        "alpha_topo": alpha_topo,
    }


@torch.no_grad()
def evaluate(model, loaders, device, topo_fn=None, topo_grid=32):
    model.eval()
    metrics = {}
    for name, loader in loaders.items():
        mse_sum = 0.0
        topo_sum = 0.0
        n = 0
        for batch in loader:
            p, truth = batch[0], batch[1]
            p = p.to(device)
            truth = truth.to(device)
            pred = model(p) if not hasattr(model, "stab_head") else model(p, return_stab=False)
            mse_sum += F.mse_loss(pred, truth, reduction="sum").item()
            if topo_fn is not None:
                pred_s = F.adaptive_avg_pool2d(pred, (topo_grid, topo_grid))
                truth_s = F.adaptive_avg_pool2d(truth, (topo_grid, topo_grid))
                topo_sum += topo_fn(pred_s, truth_s).item() * p.shape[0]
            n += p.shape[0]
        metrics[f"{name}_mse"] = mse_sum / max(n, 1)
        if topo_fn is not None:
            metrics[f"{name}_topodist"] = topo_sum / max(n, 1)
    return metrics


def train(model, dataset, config):
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    run_dir = config["run_dir"]
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "figs"), exist_ok=True)
    train_loader, val_loader, bound_loader, data, splits = dataset
    loaders = {"val": val_loader, "boundary": bound_loader}
    meta = data.get("meta", {})
    lr = float(config.get("lr", 1e-3))
    wd = float(config.get("weight_decay", 0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    use_amp = config.get("mixed_precision", False) and device.type == "cuda"
    if use_amp:
        scaler = torch.amp.GradScaler("cuda")
    else:
        class _DummyScaler:
            def scale(self, x): return x
            def step(self, o): o.step()
            def update(self): pass
        scaler = _DummyScaler()

    topo_fn = None
    if config.get("alpha_topo", 0) > 0:
        topo_grid = config.get("topo_grid", 32)
        topo_fn = TopologyLoss(grid_size=topo_grid, beta=config.get("beta_topo", 1.0))

    epochs = config.get("epochs", 10)
    loss_eq = "L = MSE + lambda_boundary*L_boundary + lambda_gamma*L_gamma + lambda_aux*L_aux + alpha*L_topo"
    train_mses = []
    val_mses = []
    topo_val_weight = config.get("topo_val_weight", 0.0)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    with open(metrics_path, "w") as mf:
        mf.write(json.dumps({"loss_equation": loss_eq}) + "\n")
        best_val = float("inf")
        for ep in range(epochs):
            t0 = time.perf_counter()
            out = train_one_epoch(model, {"train": train_loader}, device, optimizer, scaler, config, topo_fn=topo_fn, meta=meta, epoch=ep)
            train_mse = out["train_mse"]
            train_mses.append(train_mse)
            # Print exact loss formula: total = L_spectral + lambda_boundary*L_boundary + lambda_aux*L_aux + lambda_gamma*L_gamma + alpha_topo*L_topo
            lb, la, lg, at = out.get("lambda_boundary", 0), out.get("lambda_aux", 0), out.get("lambda_gamma", 0), out.get("alpha_topo", 0)
            print(f"  epoch {ep}: L_spectral={out['train_mse']:.6g} L_boundary={out['mse_boundary']:.6g} L_aux={out['aux_loss']:.6g} L_gamma={out['L_gamma']:.6g} L_topo={out['topo_loss']:.6g}")
            print(f"           lambda_boundary={lb} lambda_aux={la} lambda_gamma={lg} alpha_topo={at}")
            print(f"           computed_sum={out['computed_sum']:.6g} total_loss={out['total_loss']:.6g} (must match) grad_norm={out['grad_norm']:.6g}")
            if ep == 0 and out["aux_loss"] > 0.1:
                print("           (Total dominated by L_aux; to make topology drive learning, increase alpha_topo or reduce lambda_aux.)")
            ev = evaluate(model, loaders, device, topo_fn=topo_fn, topo_grid=config.get("topo_grid", 32))
            t1 = time.perf_counter()
            v_mse = ev["val_mse"]
            val_mses.append(v_mse)
            # For topology-trained models: select checkpoint by spectral + topology (not just MSE)
            if topo_val_weight > 0 and topo_fn is not None and "val_topodist" in ev:
                v_topo = ev.get("val_topodist", 0.0)
                objective = v_mse + topo_val_weight * v_topo
            else:
                objective = v_mse
            if objective < best_val:
                best_val = objective
                torch.save({"model": model.state_dict(), "epoch": ep}, os.path.join(run_dir, "checkpoint.pt"))
            rec = {"epoch": ep, "train_mse": train_mse, "val_mse": v_mse, "boundary_mse": ev.get("boundary_mse", 0), "val_topodist": ev.get("val_topodist", 0), "runtime": t1 - t0}
            if config.get("alpha_topo", 0) > 0 and ep == 0:
                rec["grad_norm_topo"] = out.get("grad_norm", 0)
            mf.write(json.dumps(rec) + "\n")
            mf.flush()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(train_mses, label="train_mse")
    plt.plot(val_mses, label="val_mse")
    plt.legend()
    plt.savefig(os.path.join(run_dir, "train_curve.png"))
    plt.close()
    plt.figure()
    plt.plot(val_mses, label="val_mse")
    plt.legend()
    plt.savefig(os.path.join(run_dir, "val_curve.png"))
    plt.close()

    summary = {
        "best_val_mse": best_val, "final_train_mse": train_mses[-1], "epochs": epochs, "loss_equation": loss_eq,
        "torch_version": str(torch.__version__), "cudnn_deterministic": torch.backends.cudnn.deterministic if torch.cuda.is_available() else None,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return {"run_dir": run_dir, "checkpoint": os.path.join(run_dir, "checkpoint.pt"), "summary": summary}
