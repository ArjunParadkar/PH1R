"""
Linearized coil-current -> plasma-parameter model for a DIII-D-like tokamak.

Plasma parameters (all normalised to [-1, 1], matching the d3d_real dataset):
  p[0]  beta_N_norm   Normalised beta.   Primary NTM drive.
  p[1]  q95_norm      Safety factor.     Controls mode rotation frequency.
  p[2]  n_e_norm      Line-averaged density.  Modulates turbulence level.
  p[3]  f_off         TAE frequency offset (energetic-particle content proxy).
  p[4]  amp_sc        Mode amplitude scale factor.
  p[5]  coup          NTM harmonic coupling coefficient.
  p[6]  broad         Broadband turbulence amplitude.
  p[7]  stoch         Stochastic offset to gamma_max.

Coil control inputs (all normalised to [-1, 1]):
  u[0]  I_TF   Toroidal field coil current.
  u[1]  I_CS   Central solenoid current  (drives plasma current I_p).
  u[2]  I_PF1  Poloidal field coil #1   (upper inner, mainly controls kappa).
  u[3]  I_PF2  Poloidal field coil #2   (upper outer).
  u[4]  I_PF3  Poloidal field coil #3   (lower outer).
  u[5]  I_PF4  Poloidal field coil #4   (lower inner, mainly controls delta).
  u[6]  I_EFC1 Error-field-correction coil #1  (n=1 dominant error field).
  u[7]  I_EFC2 Error-field-correction coil #2  (n=2 / higher harmonics).

Physics basis
-------------
TF coils (I_TF):
  B_T ~ I_TF.  beta_N = beta * B_T*a/(I_p/1e6) ∝ p / (I_TF^2) so
  increasing I_TF LOWERS beta_N (stabilising) and RAISES q95 (I_TF ↑ -> q95 ↑).

Central solenoid (I_CS):
  I_CS drives plasma current I_p.  q95 ~ B_T*a^2 / (R*I_p) so
  increasing I_CS LOWERS q95.  Higher I_p also mildly raises n_e.

PF coils:
  PF1+PF2 primarily control elongation kappa -> couples into n_e (density
  profile shape) and mild coupling to beta_N through pressure balance.
  PF3+PF4 primarily control triangularity delta -> modifies rational-surface
  positions and NTM mode frequency (f_off coupling).

EFC coils:
  EFC1: cancels the dominant n=1 external error field that seeds NTM modes.
    Directly reduces the mode amplitude (amp_sc).
  EFC2: cancels n=2 harmonics -> reduces harmonic coupling (coup) and
    broadband turbulence excitation (broad).

Jacobian J (8x8, plasma params x coil currents)
  p_new = p_current + J @ delta_u
The Jacobian is derived from the linearised physics above with empirical gains
representative of DIII-D operational experience.  Rows are plasma params,
columns are coil currents [TF, CS, PF1, PF2, PF3, PF4, EFC1, EFC2].
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Jacobian: rows = plasma params [0..7], cols = coil currents [0..7]
# Sign convention: positive entry means increasing that coil current
# raises that plasma parameter.
# ---------------------------------------------------------------------------
#               I_TF   I_CS   I_PF1  I_PF2  I_PF3  I_PF4  I_EFC1 I_EFC2
_J_NOMINAL = np.array([
    [-0.60,  0.00,  -0.10, -0.05,  0.05,  0.10,  0.00,  0.00],  # beta_N
    [ 0.50, -0.55,   0.05,  0.05,  0.05,  0.05,  0.00,  0.00],  # q95
    [ 0.00,  0.15,   0.20,  0.10, -0.10, -0.20,  0.00,  0.00],  # n_e
    [ 0.00,  0.00,   0.00,  0.05, -0.05,  0.00,  0.10,  0.05],  # f_off
    [ 0.00,  0.00,   0.00,  0.00,  0.00,  0.00, -0.50, -0.10],  # amp_sc
    [ 0.00,  0.00,   0.00,  0.00,  0.00,  0.00, -0.10, -0.40],  # coup
    [ 0.00,  0.05,   0.00,  0.00,  0.00,  0.00, -0.05, -0.25],  # broad
    [ 0.00,  0.00,   0.00,  0.00,  0.00,  0.00,  0.00,  0.00],  # stoch (noise)
], dtype=np.float32)

# Human-readable names for the 8 coil inputs
COIL_NAMES = ["TF", "CS", "PF1", "PF2", "PF3", "PF4", "EFC1", "EFC2"]
PARAM_NAMES = ["beta_N", "q95", "n_e", "f_off", "amp_sc", "coup", "broad", "stoch"]

# Physical actuator rate limits (normalised units per control step)
# Represent ramp-rate constraints on power supplies
COIL_RATE_LIMITS = np.array([0.05, 0.08, 0.10, 0.10, 0.10, 0.10, 0.15, 0.15],
                             dtype=np.float32)

# Absolute current limits (normalised units)
COIL_ABS_LIMITS = np.ones(8, dtype=np.float32)


class LinearizedCoilModel:
    """
    Maps coil current adjustments delta_u to plasma parameter changes delta_p.

    Usage
    -----
    model = LinearizedCoilModel()
    p_new = model.step(p_current, delta_u)   # numpy arrays [8]
    J = model.jacobian_tensor(device)         # torch tensor [8, 8]
    """

    def __init__(self, J: np.ndarray = None, noise_std: float = 0.005):
        """
        Parameters
        ----------
        J : ndarray [8, 8], optional
            Custom Jacobian.  Defaults to _J_NOMINAL.
        noise_std : float
            Standard deviation of Gaussian noise added to each step to model
            real plasma turbulence and unmodelled dynamics.
        """
        self.J = J if J is not None else _J_NOMINAL.copy()
        self.noise_std = noise_std
        self._u = np.zeros(8, dtype=np.float32)   # current coil setpoints

    @property
    def n_coils(self):
        return 8

    @property
    def n_params(self):
        return 8

    def reset(self, u_init: np.ndarray = None):
        """Reset coil setpoints (e.g. at the start of a shot)."""
        self._u = np.zeros(8, dtype=np.float32) if u_init is None else u_init.copy()

    def apply(
        self,
        p_current: np.ndarray,
        delta_u: np.ndarray,
        rng: np.random.Generator = None,
        add_noise: bool = True,
    ) -> np.ndarray:
        """
        Advance the plasma state by applying coil adjustments delta_u.

        Parameters
        ----------
        p_current : ndarray [8]
            Current normalised plasma parameters.
        delta_u : ndarray [8]
            Requested coil current adjustments.
        rng : np.random.Generator, optional
        add_noise : bool
            Whether to add stochastic noise (simulates unmodelled dynamics).

        Returns
        -------
        p_new : ndarray [8]
            Updated plasma parameters (clipped to [-1.2, 1.2]).
        delta_u_applied : ndarray [8]
            Actual delta_u after rate-limit saturation.
        """
        # Saturate at rate limits
        delta_u_sat = np.clip(delta_u, -COIL_RATE_LIMITS, COIL_RATE_LIMITS)
        # Update coil setpoints and saturate at absolute limits
        self._u = np.clip(self._u + delta_u_sat, -COIL_ABS_LIMITS, COIL_ABS_LIMITS)
        # Map to plasma parameter change
        delta_p = self.J @ self._u    # full setpoint effect (not incremental)
        p_new = p_current + 0.1 * delta_p  # 0.1: gain for one control timestep
        # Stochastic component: plasma fluctuations and unmodelled dynamics
        if add_noise:
            _rng = rng if rng is not None else np.random.default_rng()
            noise = _rng.normal(0, self.noise_std, size=8).astype(np.float32)
            noise[7] = 0.0   # stoch param updated separately
            p_new = p_new + noise
        return np.clip(p_new, -1.2, 1.2).astype(np.float32), delta_u_sat

    def jacobian_tensor(self, device: torch.device = None) -> torch.Tensor:
        """Return the Jacobian J as a torch tensor for use in gradient-based control."""
        t = torch.from_numpy(self.J)
        return t.to(device) if device is not None else t
