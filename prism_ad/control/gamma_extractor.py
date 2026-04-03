"""
Differentiable gamma_max extractor from a 2-D density grid.

The TopoFNO output is a [B, 1, H, W] density image on the (omega, gamma) plane
where the rows correspond to omega (real frequency) and columns correspond to
gamma (growth rate).  The dataset convention (see meta.json) is:

    axis 0 (rows, height H) -> omega in [omega_min, omega_max]
    axis 1 (cols, width  W) -> gamma in [gamma_min, gamma_max]

gamma_max is the dominant growth rate: the gamma value associated with the
highest-density spectral peak.  We offer three estimators, all differentiable:

1. SoftArgmax (default, recommended):
   Compute the marginal density along the gamma axis, then take the
   soft-argmax (temperature-weighted expectation):

       rho_gamma(j) = sum_i rho(i, j)          [marginal over omega]
       gamma_pred   = sum_j gamma_j * softmax(rho_gamma / tau)[j]

   With tau -> 0 this recovers the exact argmax; with tau > 0 it is smooth
   and differentiable everywhere.  We use tau = 0.05 (tunable).

2. WeightedMean:
   gamma_pred = sum_{i,j} gamma_j * rho(i, j)  [full 2-D weighted mean]
   This gives the mean growth rate across ALL eigenvalues, not the maximum.
   Useful as a stability margin metric but underestimates gamma_max.

3. Percentile (non-differentiable, for evaluation only):
   gamma_pred = percentile(rho, 99) projected onto gamma axis.

The SoftArgmax estimator is what the MPC controller uses during gradient
optimisation.  The gradient flows cleanly through it and into the TopoFNO
weights (though in inference/control we freeze the model weights and only
optimise over the coil currents / plasma parameters).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableGammaMax(nn.Module):
    """
    Extract a differentiable estimate of gamma_max from a density grid.

    Parameters
    ----------
    gamma_min, gamma_max : float
        Physical bounds of the gamma axis in the grid (from meta.json).
        Defaults match d3d_real / fusion_real_tcv datasets.
    grid_size : int
        Number of pixels along each axis (default 128).
    tau : float
        Softmax temperature.  Lower = sharper (closer to true argmax).
    method : str
        'softargmax' (default) | 'weighted_mean'
    """

    def __init__(
        self,
        gamma_min: float = -0.3,
        gamma_max: float = 0.3,
        grid_size: int = 128,
        tau: float = 0.05,
        method: str = "softargmax",
    ):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.grid_size = grid_size
        self.tau = tau
        self.method = method

        # Pre-compute the gamma axis (not a parameter — fixed by the dataset)
        gamma_vals = torch.linspace(gamma_min, gamma_max, grid_size)   # [W]
        self.register_buffer("gamma_vals", gamma_vals)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        rho : Tensor [B, 1, H, W]
            Normalised density grid output from TopoFNO (sum-to-one).

        Returns
        -------
        gamma_pred : Tensor [B]
            Per-sample estimated gamma_max, differentiable w.r.t. rho.
        """
        # rho: [B, 1, H, W] -> squeeze channel: [B, H, W]
        r = rho.squeeze(1)

        if self.method == "weighted_mean":
            # Full 2-D expectation: E[gamma] = sum_{i,j} gamma_j * rho_{ij}
            # gamma_vals: [W], r: [B, H, W]
            g = self.gamma_vals.to(r.device)   # [W]
            gamma_pred = (r * g.unsqueeze(0).unsqueeze(0)).sum(dim=(1, 2))
            return gamma_pred

        # softargmax (default)
        # 1. Marginal density along gamma axis: sum over omega rows
        rho_gamma = r.sum(dim=1)               # [B, W]
        # 2. Softmax-weighted expectation (soft argmax)
        g = self.gamma_vals.to(r.device)       # [W]
        weights = F.softmax(rho_gamma / self.tau, dim=1)   # [B, W]
        gamma_pred = (weights * g.unsqueeze(0)).sum(dim=1)  # [B]
        return gamma_pred

    def stability_margin(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Stability margin = -gamma_max.  Positive means stable.
        Useful as the objective to maximise (or equivalently, gamma_max to minimise).
        """
        return -self.forward(rho)


class InstabilityIndex(nn.Module):
    """
    Composite instability index combining gamma_max with topological features.

    Index = gamma_max + beta_topo * TopoDist(rho, rho_stable_ref)

    where rho_stable_ref is a reference stable spectrum (pure background noise,
    typically a single-peaked or flat distribution with gamma_max < 0).

    This index penalises both high growth rate AND topological deviation from
    the stable reference, making the controller more conservative near the
    boundary (where gamma_max ~ 0 but topology changes).

    For the control demo, beta_topo = 0 (we only optimise gamma_max) unless
    a stable reference is provided.
    """

    def __init__(self, gamma_extractor: DifferentiableGammaMax, beta_topo: float = 0.0):
        super().__init__()
        self.gamma_extractor = gamma_extractor
        self.beta_topo = beta_topo
        self._ref = None   # [1, 1, H, W] stable reference spectrum

    def set_stable_reference(self, rho_ref: torch.Tensor):
        """Set the reference stable spectrum for topology penalty."""
        self._ref = rho_ref.detach()

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """Returns instability index [B], differentiable."""
        idx = self.gamma_extractor(rho)
        if self.beta_topo > 0.0 and self._ref is not None:
            ref = self._ref.to(rho.device).expand_as(rho)
            topo_dist = F.mse_loss(rho, ref, reduction="none").flatten(1).sum(1)
            idx = idx + self.beta_topo * topo_dist
        return idx
