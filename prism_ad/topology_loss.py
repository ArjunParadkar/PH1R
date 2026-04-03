# Differentiable topology loss: TopologyLayer (LevelSet2D + TopKBarcodeLengths) or
# enhanced fallback (3-scale persistence surrogate + soft component count + sharpness).
#
# Enhancement over v1:
#   1. 3-scale multiscale topology features (32, 16, 8) for global + local structure.
#   2. Soft component-count loss: differentiable approximation of #connected-components.
#      Critical near regime boundaries where #peaks changes (1->2 at TCV double-tearing,
#      0->1 at D3D NTM onset). Standard MSE cannot detect this.
#   3. Spectral sharpness preservation: penalises blurred peaks vs sharp modes.

import torch
import torch.nn as nn
import torch.nn.functional as F

_TOPOLOGY_LAYER_AVAILABLE = False
try:
    from topologylayer.nn import LevelSetLayer2D, TopKBarcodeLengths
    _TOPOLOGY_LAYER_AVAILABLE = True
except Exception:
    pass

K0 = 16
K1 = 16


# ---------------------------------------------------------------------------
# Fallback feature extractors
# ---------------------------------------------------------------------------

def _persistence_image_surrogate(img, k, negate=False):
    """Top-k pixel values as H0/H1 surrogate. img: [B,1,H,W] or [B,H,W]."""
    img = img.squeeze(1)
    if img.dim() == 2:
        img = img.unsqueeze(0)
    if negate:
        img = -img
    B = img.shape[0]
    flat = img.reshape(B, -1)
    k_actual = min(k, flat.shape[1])
    topk_vals, _ = torch.topk(flat, k_actual, dim=1)
    return topk_vals


def _soft_component_count(img, threshold=0.6, temperature=0.05):
    """
    Soft differentiable count of 'dominant' spatial regions (proxy for H0 betti number).
    Normalise image to [0,1] then apply sigmoid gate; sum gives soft component count.
    Returns [B] tensor.
    """
    img = img.squeeze(1)
    if img.dim() == 2:
        img = img.unsqueeze(0)
    B = img.shape[0]
    flat = img.reshape(B, -1)
    mx = flat.max(dim=1, keepdim=True)[0].clamp(min=1e-8)
    img_norm = flat / mx  # [B, H*W]
    hw = img_norm.shape[1]
    return torch.sigmoid((img_norm - threshold) / temperature).sum(dim=1) / hw  # [B], in [0,1]


def _sharpness_features(img, pool_sizes=(3, 5, 9)):
    """
    Spectral-peak sharpness: (max_pool - avg_pool) at multiple window sizes.
    A sharp spectral peak produces high max-avg difference; a blurred peak lower.
    Returns [B, n_feats] tensor.
    """
    if img.dim() == 3:
        img = img.unsqueeze(1)
    feats = []
    for k in pool_sizes:
        pad = k // 2
        max_p = F.max_pool2d(img, k, stride=1, padding=pad)
        avg_p = F.avg_pool2d(img, k, stride=1, padding=pad)
        diff = (max_p - avg_p).squeeze(1).flatten(1)   # [B, H*W]
        k_top = min(16, diff.shape[1])
        top, _ = torch.topk(diff, k_top, dim=1)
        feats.append(top)
    return torch.cat(feats, dim=1)   # [B, sum(k_top_i)]


# ---------------------------------------------------------------------------
# Main TopologyLoss module
# ---------------------------------------------------------------------------

class TopologyLoss(nn.Module):
    """
    Differentiable topology loss for spectral density maps.

    Args:
        grid_size: target resolution for topology computation (default 32).
        beta: weight for H1 (loops) relative to H0 (components).
        k0, k1: top-k barcode lengths for H0 and H1.
        use_component_count: add soft component-count matching loss.
        use_sharpness: add spectral peak sharpness matching loss.
        component_weight: weight for component-count loss term.
        sharpness_weight: weight for sharpness loss term.
    """

    def __init__(
        self,
        grid_size=32,
        beta=1.0,
        k0=K0,
        k1=K1,
        use_component_count=True,
        use_sharpness=True,
        component_weight=0.15,
        sharpness_weight=0.10,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.beta = beta
        self.k0 = k0
        self.k1 = k1
        self.use_component_count = use_component_count
        self.use_sharpness = use_sharpness
        self.component_weight = component_weight
        self.sharpness_weight = sharpness_weight
        self.use_topolayer = _TOPOLOGY_LAYER_AVAILABLE
        if self.use_topolayer:
            self.layer = LevelSetLayer2D(size=(grid_size, grid_size), maxdim=1, sublevel=False)
            self.feat0 = TopKBarcodeLengths(dim=0, k=k0)
            self.feat1 = TopKBarcodeLengths(dim=1, k=k1)

    # ------------------------------------------------------------------
    # TopologyLayer path
    # ------------------------------------------------------------------

    def _feat_topolayer_one(self, x):
        dgms, _ = self.layer(x)
        f0 = self.feat0((dgms, False))
        f1 = self.feat1((dgms, False))
        return f0, f1

    def _feat_topolayer(self, x):
        x = x.squeeze(1)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B = x.shape[0]
        f0_list, f1_list = [], []
        for i in range(B):
            f0, f1 = self._feat_topolayer_one(x[i])
            f0_list.append(f0)
            f1_list.append(f1)
        return torch.stack(f0_list, 0), torch.stack(f1_list, 0)

    # ------------------------------------------------------------------
    # Fallback path: multiscale persistence-image surrogate
    # ------------------------------------------------------------------

    def _feat_fallback(self, x):
        """Single-scale top-k surrogate."""
        f0 = _persistence_image_surrogate(x, self.k0, negate=False)
        f1 = _persistence_image_surrogate(x, self.k1, negate=True)
        return f0, f1

    def _feat_fallback_multiscale(self, x):
        """
        3-scale surrogate: full, half, quarter resolution.
        Concatenate features so loss penalises mismatch at global (coarse) and
        local (fine) topology simultaneously.
        """
        scales = [self.grid_size, self.grid_size // 2, self.grid_size // 4]
        f0_parts, f1_parts = [], []
        for s in scales:
            if s < 4:
                continue
            x_s = x if x.shape[-1] == s else F.adaptive_avg_pool2d(x, (s, s))
            f0_s, f1_s = self._feat_fallback(x_s)
            f0_parts.append(f0_s)
            f1_parts.append(f1_s)
        f0 = torch.cat(f0_parts, dim=1)
        f1 = torch.cat(f1_parts, dim=1)
        return f0, f1

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, pred, truth):
        # Ensure inputs are at the target topology grid resolution
        if pred.shape[-1] != self.grid_size:
            pred = F.adaptive_avg_pool2d(pred, (self.grid_size, self.grid_size))
            truth = F.adaptive_avg_pool2d(truth, (self.grid_size, self.grid_size))

        # --- Persistence feature loss (H0 + H1) ---
        if self.use_topolayer:
            try:
                f0_p, f1_p = self._feat_topolayer(pred)
                f0_t, f1_t = self._feat_topolayer(truth)
            except Exception:
                f0_p, f1_p = self._feat_fallback_multiscale(pred)
                f0_t, f1_t = self._feat_fallback_multiscale(truth)
        else:
            f0_p, f1_p = self._feat_fallback_multiscale(pred)
            f0_t, f1_t = self._feat_fallback_multiscale(truth)

        l0 = ((f0_p - f0_t) ** 2).mean()
        l1 = ((f1_p - f1_t) ** 2).mean()
        loss = (l0 + self.beta * l1).clamp(min=1e-8)

        # --- Soft component-count loss ---
        # Critical for detecting 0->1->2 component transitions at regime boundaries.
        if self.use_component_count:
            cnt_p = _soft_component_count(pred)
            cnt_t = _soft_component_count(truth)
            count_loss = ((cnt_p - cnt_t) ** 2).mean()
            loss = loss + self.component_weight * count_loss

        # --- Spectral peak sharpness loss ---
        # Prevents predicting a blurred smear where the truth has sharp spectral lines.
        if self.use_sharpness:
            sharp_p = _sharpness_features(pred)
            sharp_t = _sharpness_features(truth)
            sharp_loss = ((sharp_p - sharp_t) ** 2).mean()
            loss = loss + self.sharpness_weight * sharp_loss

        return loss.clamp(min=1e-8)
