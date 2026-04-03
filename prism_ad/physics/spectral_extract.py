# Extract (omega, gamma) per mode from time series: linear fit for gamma, FFT peak for omega.

import numpy as np


def _linear_window(t, signal, frac=0.3, min_pts=10):
    logmag = np.log(np.abs(signal) + 1e-12)
    n = len(t)
    win = max(min_pts, int(n * frac))
    if win >= n:
        return 0, n
    mid = n // 2
    return max(0, mid - win // 2), min(n, mid + win // 2)


def estimate_gamma_from_log_amplitude(t, amplitude, window_frac=0.3):
    """Estimate growth rate gamma by linear fit of log|E(t)| in a linear window."""
    t = np.asarray(t).flatten()
    amplitude = np.asarray(amplitude).flatten()
    if len(t) < 5 or len(t) != len(amplitude):
        return 0.0
    logmag = np.log(np.abs(amplitude) + 1e-12)
    i0, i1 = _linear_window(t, amplitude, frac=window_frac)
    if i1 - i0 < 3:
        return 0.0
    t_win = t[i0:i1]
    y_win = logmag[i0:i1]
    try:
        coeffs = np.polyfit(t_win, y_win, 1)
        return float(coeffs[0])
    except Exception:
        return 0.0


def estimate_omega_from_fft(t, signal, window_frac=0.3):
    """Estimate frequency omega via FFT peak in the same linear window."""
    t = np.asarray(t).flatten()
    signal = np.asarray(signal).flatten()
    if len(t) < 10 or len(t) != len(signal):
        return 0.0
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    i0, i1 = _linear_window(t, signal, frac=window_frac)
    seg = signal[i0:i1]
    if len(seg) < 4:
        return 0.0
    n = len(seg)
    fft = np.fft.rfft(seg)
    freqs = np.fft.rfftfreq(n, dt)
    idx = np.argmax(np.abs(fft)[1:]) + 1
    return float(2 * np.pi * freqs[idx])


def extract_eigenvalue_cloud_from_modes(t_list, amplitude_list, window_frac=0.3):
    """
    For each mode: gamma_k from log|E_k(t)| linear fit, omega_k from FFT peak.
    Returns complex array lambda_k = omega_k + i*gamma_k and gamma_max.
    """
    out = []
    for t, amp in zip(t_list, amplitude_list):
        t = np.asarray(t)
        amp = np.asarray(amp)
        g = estimate_gamma_from_log_amplitude(t, amp, window_frac=window_frac)
        o = estimate_omega_from_fft(t, amp, window_frac=window_frac)
        out.append(o + 1j * g)
    out = np.array(out, dtype=np.complex128)
    gamma_max = float(np.max(out.imag)) if out.size else 0.0
    return out, gamma_max
