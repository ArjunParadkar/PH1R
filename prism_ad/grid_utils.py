# Fixed (omega, gamma) density grid: Gaussian kernel in physical units, 128x128, sum-to-1 after mapping.

import numpy as np


def eigs_to_density(eigs, grid_size, omega_min, omega_max, gamma_min, gamma_max, sigma):
    """
    sigma is in physical (omega, gamma) coordinate units, not pixels.
    Eigenvalues outside [omega_min, omega_max] x [gamma_min, gamma_max] are discarded; grid is then renormalized.
    """
    w = np.linspace(omega_min, omega_max, grid_size)
    g = np.linspace(gamma_min, gamma_max, grid_size)
    omega_g, gamma_g = np.meshgrid(w, g)
    grid = np.zeros((grid_size, grid_size), dtype=np.float64)
    e = np.asarray(eigs).flatten()
    e = e[np.isfinite(e)]
    in_domain = (
        (e.real >= omega_min) & (e.real <= omega_max) &
        (e.imag >= gamma_min) & (e.imag <= gamma_max)
    )
    e = e[in_domain]
    if e.size == 0:
        grid[grid_size // 2, grid_size // 2] = 1.0
        return grid.astype(np.float32)
    for lam in e:
        o, gm = lam.real, lam.imag
        grid += np.exp(-((omega_g - o) ** 2 + (gamma_g - gm) ** 2) / (2 * sigma ** 2))
    s = grid.sum()
    if s > 0:
        grid = grid / s
    return grid.astype(np.float32)
