# MLP, CNN, FNO, PI-FNO, Topo-FNO: params -> [B,1,128,128]; softplus + sum-to-1.
#
# TopoFNO enhancements over v1:
#   - 6 FNO layers, 24 channels, 20 Fourier modes (vs 4/16/16)
#   - Inherits from PIFNO: includes physics stability head (lambda_aux in training)
#   - SpectralAttentionGate: learns spatial attention on the spectral feature map,
#     focusing on dominant mode locations (peaks change topology at boundaries)
#   - TopologyRefinementBlock: residual 3-layer CNN that enhances local peak structure
#     after main FNO body; trained with topology loss to preserve spectral components

import math
import torch
import torch.nn as nn


def normalize_sum_to_one(x):
    """rho_pred = softplus(z); rho_pred = rho_pred / rho_pred.sum() so predictions match dataset distribution."""
    s = x.flatten(1).sum(1, keepdim=True).clamp(min=1e-8)
    return x / s.view(-1, 1, 1, 1)


class MLP(nn.Module):
    def __init__(self, param_dim, grid_size=128):
        super().__init__()
        self.param_dim = param_dim
        self.grid_size = grid_size
        self.fc = nn.Sequential(
            nn.Linear(param_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, grid_size * grid_size),
        )

    def forward(self, p):
        x = self.fc(p)
        x = x.view(-1, 1, self.grid_size, self.grid_size)
        x = nn.functional.softplus(x)
        return normalize_sum_to_one(x)


class CNN(nn.Module):
    def __init__(self, param_dim, grid_size=128):
        super().__init__()
        self.grid_size = grid_size
        self.embed = nn.Linear(param_dim, 256)
        self.proj = nn.Linear(256, 64 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(8, 1, 4, 2, 1),
        )

    def forward(self, p):
        z = nn.functional.gelu(self.embed(p))
        z = self.proj(z).view(-1, 64, 8, 8)
        x = self.decoder(z)
        if x.shape[-1] != self.grid_size:
            x = nn.functional.interpolate(x, size=(self.grid_size, self.grid_size), mode="bilinear", align_corners=False)
        x = nn.functional.softplus(x)
        return normalize_sum_to_one(x)


class FourierBlock2d(nn.Module):
    def __init__(self, channels, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / (channels * channels)
        self.weights = nn.Parameter(torch.randn(channels, channels, modes1, modes2, 2) * 0.1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros_like(x_ft)
        m1, m2 = min(self.modes1, H // 2), min(self.modes2, W // 2 + 1)
        w = self.weights[:, :, :m1, :m2]
        w = torch.view_as_complex(w)
        out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], w)
        out_ft[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], w)
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNO2d(nn.Module):
    def __init__(self, param_dim, grid_size=128, channels=16, modes=16, n_layers=4):
        super().__init__()
        self.grid_size = grid_size
        self.channels = channels
        self.embed = nn.Linear(param_dim, channels)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "fourier": FourierBlock2d(channels, modes, modes),
                "pw": nn.Conv2d(channels, channels, 1),
            })
            for _ in range(n_layers)
        ])
        self.final = nn.Conv2d(channels + 2, 1, 1)

    def forward(self, p):
        B = p.shape[0]
        e = self.embed(p)
        h = e.view(B, -1, 1, 1).expand(-1, -1, self.grid_size, self.grid_size)
        xs = torch.linspace(-1, 1, self.grid_size, device=p.device, dtype=p.dtype)
        xg, yg = torch.meshgrid(xs, xs, indexing="ij")
        coord = torch.stack([xg, yg], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        h = torch.cat([h, coord], dim=1)
        for blk in self.blocks:
            c = h.shape[1]
            h_f = h[:, : self.channels]
            h_f = blk["fourier"](h_f) + blk["pw"](h_f)
            h_f = nn.functional.gelu(h_f)
            h = torch.cat([h_f, h[:, self.channels :]], dim=1)
        h = self.final(h)
        h = nn.functional.softplus(h)
        return normalize_sum_to_one(h)


class PIFNO(FNO2d):
    def __init__(self, param_dim, grid_size=128, channels=16, modes=16, n_layers=4, n_stab_classes=3):
        super().__init__(param_dim, grid_size, channels, modes, n_layers)
        self.embed = nn.Linear(param_dim, channels)
        self.stab_head = nn.Linear(channels, n_stab_classes)

    def forward(self, p, return_stab=False):
        B = p.shape[0]
        e = self.embed(p)
        h = e.view(B, -1, 1, 1).expand(-1, -1, self.grid_size, self.grid_size)
        xs = torch.linspace(-1, 1, self.grid_size, device=p.device, dtype=p.dtype)
        xg, yg = torch.meshgrid(xs, xs, indexing="ij")
        coord = torch.stack([xg, yg], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        h = torch.cat([h, coord], dim=1)
        for blk in self.blocks:
            h_f = h[:, : self.channels]
            h_f = blk["fourier"](h_f) + blk["pw"](h_f)
            h_f = nn.functional.gelu(h_f)
            h = torch.cat([h_f, h[:, self.channels :]], dim=1)
        out = self.final(h)
        out = nn.functional.softplus(out)
        out = normalize_sum_to_one(out)
        y_stab = self.stab_head(e)
        if return_stab:
            return out, y_stab
        return out


class _SpectralAttentionGate(nn.Module):
    """
    Learns a spatial attention map over the FNO feature space.
    Emphasises locations of dominant spectral modes (peaks that change topology
    at stability boundaries). Computed from the full h = [FNO_features | coords].
    Output: soft gate in [0, 1] applied as out_gated = out * (1 + gate).
    """
    def __init__(self, in_channels):
        super().__init__()
        mid = max(in_channels // 2, 8)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, h):
        return self.net(h)   # [B, 1, H, W]


class _TopologyRefinementBlock(nn.Module):
    """
    Residual CNN applied after the main FNO output.
    Sharpens spectral peaks and suppresses artefacts that violate topological
    structure (spurious secondary peaks, missing components).
    Trained with topology loss; at inference this block runs fast (3 conv layers).
    """
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x):
        return x + self.net(x)   # residual: preserve main prediction, only refine


class TopoFNO(PIFNO):
    """
    Topology-Aware FNO (enhanced v2).

    Architecture improvements over vanilla TopoFNO:
      - 6 Fourier layers (vs 4), 24 channels (vs 16), 20 modes (vs 16)
        -> richer spectral basis, better regime-transition representation
      - Inherits PIFNO stability head (physics constraint in training)
      - SpectralAttentionGate: spatial attention over [features+coords],
        gating the output to emphasise dominant spectral peaks
      - TopologyRefinementBlock: residual CNN to sharpen peaks and
        suppress spurious artefacts that violate topological structure

    At inference: forward pass is slightly slower than PIFNO due to the
    two extra modules, but the key claim is accuracy (not speed).
    Training: topology loss (alpha_topo * TopologyLoss) plus all PIFNO losses.
    """

    def __init__(self, param_dim, grid_size=128, channels=24, modes=20, n_layers=6, n_stab_classes=3):
        super().__init__(param_dim, grid_size, channels, modes, n_layers, n_stab_classes)
        # Spectral attention gate: takes full feature map h (channels + 2 coord channels)
        self.spectral_attn = _SpectralAttentionGate(in_channels=channels + 2)
        # Topology refinement: residual CNN on the scalar output map
        self.topo_refine = _TopologyRefinementBlock(hidden=16)
        # Scalar scale parameter (learnable; used as a mild regularisation signal)
        self._topo_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, p, return_stab=False):
        B = p.shape[0]
        e = self.embed(p)
        h = e.view(B, -1, 1, 1).expand(-1, -1, self.grid_size, self.grid_size)
        xs = torch.linspace(-1, 1, self.grid_size, device=p.device, dtype=p.dtype)
        xg, yg = torch.meshgrid(xs, xs, indexing="ij")
        coord = torch.stack([xg, yg], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        h = torch.cat([h, coord], dim=1)    # [B, channels+2, H, W]

        for blk in self.blocks:
            h_f = h[:, : self.channels]
            h_f = blk["fourier"](h_f) + blk["pw"](h_f)
            h_f = nn.functional.gelu(h_f)
            h = torch.cat([h_f, h[:, self.channels :]], dim=1)

        # Spectral attention gate: learn to focus on mode locations
        attn = self.spectral_attn(h)          # [B, 1, H, W] in [0, 1]

        # Main projection to scalar output
        out = self.final(h)                    # [B, 1, H, W]

        # Apply attention (additive gating: amplify relevant regions)
        out = out * (1.0 + attn)

        # Topology refinement (residual)
        out = self.topo_refine(out)

        out = nn.functional.softplus(out)
        out = normalize_sum_to_one(out)

        if return_stab:
            y_stab = self.stab_head(e)
            return out, y_stab
        return out
