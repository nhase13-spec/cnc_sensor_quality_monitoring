"""Signal utilities for low-rate CNC telemetry (10 Hz)."""

from __future__ import annotations

import numpy as np
from scipy import signal, stats


def rms(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(values ** 2)))


def crest_factor(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    denom = rms(values)
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float(np.max(np.abs(values)) / denom)


def spectral_features(values: np.ndarray, fs: float, min_samples: int) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size < min_samples:
        return {
            "spectral_energy": np.nan,
            "dominant_frequency_hz": np.nan,
            "spectral_centroid_hz": np.nan,
            "spectral_entropy": np.nan,
        }
    values = values - np.mean(values)
    freqs, psd = signal.welch(values, fs=fs, nperseg=min(256, values.size))
    psd = psd + 1e-12
    psd_norm = psd / psd.sum()
    centroid = float(np.sum(freqs * psd_norm))
    entropy = float(-np.sum(psd_norm * np.log(psd_norm)))
    dominant = float(freqs[int(np.argmax(psd))])
    energy = float(np.sum(psd))
    return {
        "spectral_energy": energy,
        "dominant_frequency_hz": dominant,
        "spectral_centroid_hz": centroid,
        "spectral_entropy": entropy,
    }


def wavelet_energy(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 8:
        return np.nan
    try:
        import pywt  # type: ignore
    except ImportError:
        # Fallback: Haar-like two-scale energy split without PyWavelets.
        half = values.size // 2
        approx = values[:half].mean()
        detail = values - approx
        return float(np.sum(approx ** 2) + np.sum(detail ** 2))
    coeffs = pywt.wavedec(values, "db4", level=1)
    return float(sum(np.sum(c ** 2) for c in coeffs))


def autocorr_lag1(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 3:
        return np.nan
    a = values[:-1]
    b = values[1:]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def rolling_variance_mean(values: np.ndarray, window: int = 10) -> float:
    values = values[np.isfinite(values)]
    if values.size < window:
        return np.nan
    rolling = pd_rolling_var(values, window)
    return float(np.nanmean(rolling))


def pd_rolling_var(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    for idx in range(window - 1, values.size):
        chunk = values[idx - window + 1 : idx + 1]
        out[idx] = np.var(chunk)
    return out


def trend_slope(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 3:
        return np.nan
    x = np.arange(values.size)
    slope, _, _, _, _ = stats.linregress(x, values)
    return float(slope)
