# Persistence diagrams and Wasserstein distance for evaluation (non-differentiable).

import numpy as np

_persim_available = False
_gudhi_available = False
try:
    import persim
    _persim_available = True
except Exception:
    pass
try:
    import gudhi
    _gudhi_available = True
except Exception:
    pass


def _cubical_persistence_gudhi(img):
    if not _gudhi_available:
        return None, None
    img = np.asarray(img).squeeze()
    if img.ndim != 2:
        return None, None
    try:
        cc = gudhi.CubicalComplex(dimensions=img.shape, top_dimensional_cells=(-img.ravel()).tolist())
        pers = cc.persistence() if hasattr(cc, "persistence") else cc.compute_persistence()
        dgm0 = np.array([[p[1][0], p[1][1]] for p in pers if p[0] == 0 and p[1][1] != np.inf], dtype=np.float64)
        dgm1 = np.array([[p[1][0], p[1][1]] for p in pers if p[0] == 1], dtype=np.float64)
        if dgm0.size == 0:
            dgm0 = np.zeros((0, 2))
        if dgm1.size == 0:
            dgm1 = np.zeros((0, 2))
        return dgm0, dgm1
    except Exception:
        return None, None


def get_diagrams(img):
    return _cubical_persistence_gudhi(img)


def _resize_for_gudhi(img, resolution=32):
    """Downsample a 2-D numpy array to (resolution, resolution) using block averaging."""
    from PIL import Image as _PILImage
    arr = np.asarray(img, dtype=np.float32)
    if arr.shape == (resolution, resolution):
        return arr
    # Use PIL for fast antialiased downsampling
    pil = _PILImage.fromarray(arr).resize((resolution, resolution), _PILImage.LANCZOS)
    out = np.array(pil, dtype=np.float32)
    s = out.sum()
    return out / (s + 1e-12)   # keep sum-to-one normalisation


def wasserstein_d0_d1(pred, truth, beta=1.0, resolution=32):
    pred = np.asarray(pred).squeeze()
    truth = np.asarray(truth).squeeze()
    if pred.ndim == 3:
        pred = pred[0]
    if truth.ndim == 3:
        truth = truth[0]
    pred  = _resize_for_gudhi(pred,  resolution)
    truth = _resize_for_gudhi(truth, resolution)
    d0_p, d1_p = _cubical_persistence_gudhi(pred)
    d0_t, d1_t = _cubical_persistence_gudhi(truth)
    if not _persim_available or d0_p is None:
        return float("nan"), float("nan"), float("nan")
    w0 = persim.wasserstein(d0_p, d0_t)
    w1 = persim.wasserstein(d1_p, d1_t)
    return w0, w1, w0 + beta * w1


def _l2_structure_proxy(pred, truth):
    p = np.asarray(pred).flatten().astype(np.float64)
    t = np.asarray(truth).flatten().astype(np.float64)
    p = p / (np.linalg.norm(p) + 1e-12)
    t = t / (np.linalg.norm(t) + 1e-12)
    return float(np.linalg.norm(p - t))


def topology_distance_batch(pred_batch, truth_batch, beta=1.0, max_samples=50):
    """
    Returns (w0, w1, topo_dist, structure_proxy_distance, num_H0_features, num_H1_features).
    topo_dist is real topology (W0 + beta*W1) only when GUDHI+Persim available; else None.
    structure_proxy_distance is L2 between normalized flattened images (not topology); always returned.
    num_H0_features, num_H1_features are mean counts over batch when GUDHI available; else None.
    """
    w0s, w1s, topos = [], [], []
    n0_list, n1_list = [], []
    n = min(pred_batch.shape[0], max_samples)
    structure_proxy_dists = []
    for i in range(n):
        w0, w1, td = wasserstein_d0_d1(pred_batch[i], truth_batch[i], beta=beta)
        structure_proxy_dists.append(_l2_structure_proxy(pred_batch[i], truth_batch[i]))
        if not np.isnan(td):
            w0s.append(w0)
            w1s.append(w1)
            topos.append(td)
        d0, d1 = _cubical_persistence_gudhi(pred_batch[i])
        if d0 is not None:
            n0_list.append(len(d0))
        if d1 is not None:
            n1_list.append(len(d1))
    sp = float(np.mean(structure_proxy_dists))
    n0_mean = float(np.mean(n0_list)) if n0_list else None
    n1_mean = float(np.mean(n1_list)) if n1_list else None
    if topos:
        return np.mean(w0s), np.mean(w1s), float(np.mean(topos)), sp, n0_mean, n1_mean
    return float("nan"), float("nan"), None, sp, n0_mean, n1_mean
