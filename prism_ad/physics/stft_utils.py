# Spectrogram (f,t) to 128x128 density image, sum-to-one.

import numpy as np


def stft_power(signal, fs, nperseg=256):
    """Power spectrogram S(f,t)."""
    from scipy import signal as scipy_signal
    f, t, Zxx = scipy_signal.spectrogram(signal, fs, nperseg=nperseg)
    return f, t, np.abs(Zxx) ** 2


def spectrogram_to_density_128x128(f, t, S, grid_size=128):
    """
    Resample/crop power spectrogram to grid_size x grid_size and normalize to sum=1.
    S: (n_freq, n_time). Returns density image (grid_size, grid_size) float32.
    """
    S = np.asarray(S, dtype=np.float64)
    if S.size == 0:
        out = np.zeros((grid_size, grid_size), dtype=np.float32)
        out[grid_size // 2, grid_size // 2] = 1.0
        return out
    from scipy.ndimage import zoom
    nf, nt = S.shape
    if nf < 2 or nt < 2:
        out = np.zeros((grid_size, grid_size), dtype=np.float32)
        out[min(grid_size // 2, grid_size - 1), min(grid_size // 2, grid_size - 1)] = 1.0
        return out
    zoom_f = grid_size / nf
    zoom_t = grid_size / nt
    S_resized = zoom(S, (zoom_f, zoom_t), order=1)
    S_resized = np.maximum(S_resized, 0)
    s = S_resized.sum()
    if s > 0:
        S_resized = S_resized / s
    return S_resized.astype(np.float32)
