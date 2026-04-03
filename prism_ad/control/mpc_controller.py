"""
Gradient-based Model Predictive Controller using TopoFNO as the forward model.

Key idea
--------
TopoFNO is differentiable end-to-end:

    rho_pred = TopoFNO(p)        p in R^8 (plasma params)
    gamma_pred = GammaExtractor(rho_pred)

So we can compute d(gamma_pred)/d(p) via autograd.  Given the linearised
coil model  delta_p ~ J @ delta_u,  we have:

    d(gamma)/d(delta_u) = d(gamma)/d(p) @ J

The MPC solves at each control step:

    min_{delta_u}   gamma_max(p + alpha * J @ delta_u)
                  + lambda_u * ||delta_u||^2           (actuator effort)
                  + lambda_s * ||delta_u - delta_u_prev||^2  (smoothness)
    s.t.  ||delta_u||_inf <= rate_limit

via projected gradient descent (fast, runs in <1 ms on CPU for 8 actuators).

Algorithm per control step
--------------------------
1.  Query TopoFNO: rho_now = model(p_now),  gmax_now = extractor(rho_now).
2.  If gmax_now < threshold_activate: no action (return delta_u = 0).
3.  Else: run n_opt_steps of projected gradient descent:
        a. Compute rho_opt = model(p_now + alpha * J @ delta_u_var)
        b. Compute J_cost  = extractor(rho_opt) + regularisation
        c. Backprop J_cost w.r.t. delta_u_var
        d. Update delta_u_var -= lr * grad
        e. Project: clip to [-rate_limit, +rate_limit]
4.  Return optimal delta_u.

Hyperparameters (all tunable via constructor)
--------------------------------------------
threshold_activate : float   gamma_max threshold above which control activates.
alpha              : float   Step size: how much delta_u shifts plasma params.
lambda_u           : float   L2 penalty on actuator effort (prevents saturation).
lambda_s           : float   L2 penalty on change from last action (smooth control).
n_opt_steps        : int     Gradient descent iterations per control step.
lr_opt             : float   Gradient descent learning rate.
"""

import torch
import torch.nn.functional as F
import numpy as np

from prism_ad.control.gamma_extractor import DifferentiableGammaMax, InstabilityIndex


class GradientMPCController:
    """
    Gradient-based MPC controller backed by a frozen TopoFNO model.

    Parameters
    ----------
    model : nn.Module
        Trained TopoFNO (or any model with signature rho = model(p)).
        Weights are frozen during control (only delta_u is optimised).
    coil_jacobian : Tensor [8, 8]
        Linearised Jacobian J from coil currents to plasma parameters.
    gamma_extractor : DifferentiableGammaMax
        Differentiable gamma_max estimator.
    threshold_activate : float
        Control activates only when gamma_max > this value.
    alpha : float
        Effective gain of coil action on plasma parameters per step.
    lambda_u : float
        L2 regularisation on actuator effort.
    lambda_s : float
        L2 regularisation on actuator change (smoothness).
    n_opt_steps : int
        Number of projected gradient descent steps per control query.
    lr_opt : float
        Learning rate for the inner optimisation.
    rate_limit : float
        Per-actuator rate limit (normalised units).
    device : torch.device
    """

    def __init__(
        self,
        model,
        coil_jacobian: torch.Tensor,
        gamma_extractor: DifferentiableGammaMax = None,
        threshold_activate: float = 0.02,
        alpha: float = 0.4,
        lambda_u: float = 0.05,
        lambda_s: float = 0.02,
        n_opt_steps: int = 40,
        lr_opt: float = 0.05,
        rate_limit: float = 0.15,
        device: torch.device = None,
    ):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.J = coil_jacobian.to(self.device).float()
        self.gamma_ext = (
            gamma_extractor if gamma_extractor is not None
            else DifferentiableGammaMax()
        ).to(self.device)

        self.threshold = threshold_activate
        self.alpha = alpha
        self.lambda_u = lambda_u
        self.lambda_s = lambda_s
        self.n_opt_steps = n_opt_steps
        self.lr_opt = lr_opt
        self.rate_limit = rate_limit

        self._du_prev = torch.zeros(8, device=self.device)

    def reset(self):
        self._du_prev = torch.zeros(8, device=self.device)

    def _forward_gamma(self, p_tensor: torch.Tensor, delta_u: torch.Tensor) -> torch.Tensor:
        """
        Compute predicted gamma_max after applying delta_u coil adjustments.

        p_tensor : [1, 8]
        delta_u  : [8]  (optimisation variable)
        returns  : scalar
        """
        # Linearised plasma response to coil adjustment
        delta_p = self.alpha * (self.J @ delta_u)         # [8]
        p_new = (p_tensor.squeeze(0) + delta_p).unsqueeze(0)   # [1, 8]
        p_new = p_new.clamp(-1.2, 1.2)

        # TopoFNO forward pass
        rho = self.model(p_new)
        if isinstance(rho, tuple):
            rho = rho[0]

        gamma_pred = self.gamma_ext(rho)   # [1]
        return gamma_pred.squeeze(0)

    def step(
        self,
        p_current: np.ndarray,
        verbose: bool = False,
    ) -> tuple:
        """
        Compute optimal coil adjustment for the current plasma state.

        Parameters
        ----------
        p_current : ndarray [8]
            Current normalised plasma parameters.
        verbose : bool
            Print optimisation progress.

        Returns
        -------
        delta_u : ndarray [8]
            Recommended coil current adjustments.
        info : dict
            Diagnostic info: gamma_before, gamma_after, active, opt_trace.
        """
        p_t = torch.tensor(p_current, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Evaluate current instability
        with torch.no_grad():
            rho_now = self.model(p_t)
            if isinstance(rho_now, tuple):
                rho_now = rho_now[0]
            gamma_now = self.gamma_ext(rho_now).item()

        info = {"gamma_before": gamma_now, "active": False, "opt_trace": []}

        if gamma_now <= self.threshold:
            # Stable: no intervention needed
            self._du_prev = self._du_prev * 0.9   # decay towards zero
            info["gamma_after"] = gamma_now
            return np.zeros(8, dtype=np.float32), info

        # Unstable: run projected gradient descent to find stabilising delta_u
        info["active"] = True
        delta_u_var = torch.zeros(8, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([delta_u_var], lr=self.lr_opt)

        for step_i in range(self.n_opt_steps):
            optimizer.zero_grad()
            gamma_pred = self._forward_gamma(p_t, delta_u_var)
            cost = (
                gamma_pred
                + self.lambda_u * delta_u_var.pow(2).sum()
                + self.lambda_s * (delta_u_var - self._du_prev.detach()).pow(2).sum()
            )
            cost.backward()
            optimizer.step()
            # Project: enforce rate limit
            with torch.no_grad():
                delta_u_var.clamp_(-self.rate_limit, self.rate_limit)
            if verbose and step_i % 10 == 0:
                print(f"  opt step {step_i:3d}: gamma={gamma_pred.item():.4f}  cost={cost.item():.4f}")
            info["opt_trace"].append(gamma_pred.item())

        # Final evaluation after optimisation
        with torch.no_grad():
            gamma_after = self._forward_gamma(p_t, delta_u_var).item()

        du_np = delta_u_var.detach().cpu().numpy().astype(np.float32)
        self._du_prev = delta_u_var.detach()

        info["gamma_after"] = gamma_after
        return du_np, info


class BaselineProportionalController:
    """
    Simple proportional controller for comparison.

    Acts on beta_N (p[0]) only via TF coil (u[0]): if gamma_max > 0,
    reduce beta_N by increasing I_TF.  No model knowledge — pure heuristic.
    """

    def __init__(self, gain: float = 0.5, threshold: float = 0.02):
        self.gain = gain
        self.threshold = threshold
        self._u_tf = 0.0

    def reset(self):
        self._u_tf = 0.0

    def step(self, gamma_max: float) -> np.ndarray:
        delta_u = np.zeros(8, dtype=np.float32)
        if gamma_max > self.threshold:
            # Increase TF current -> reduces beta_N
            du_tf = self.gain * gamma_max
            du_tf = float(np.clip(du_tf, 0.0, 0.15))
            delta_u[0] = du_tf
        return delta_u
