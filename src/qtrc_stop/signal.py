from __future__ import annotations

import numpy as np
from scipy import signal


def complex_from_amplitude_phase(amplitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return amplitude.astype(float) * np.exp(1j * phase.astype(float))


def robust_normalize(z: np.ndarray) -> np.ndarray:
    med_r = np.median(z.real)
    med_i = np.median(z.imag)
    centered = z - (med_r + 1j * med_i)
    scale = np.median(np.abs(centered - np.median(centered)))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = np.std(centered)
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return centered / scale


def highpass_complex(
    z: np.ndarray,
    fs: float,
    cutoff_hz: float = 50.0,
    order: int = 2,
    ripple_db: float = 0.5,
) -> np.ndarray:
    nyq = fs / 2.0
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        raise ValueError(f"Invalid cutoff {cutoff_hz} for fs={fs}")
    sos = signal.cheby1(order, ripple_db, cutoff_hz / nyq, btype="highpass", output="sos")
    padlen = min(max(3 * (2 * len(sos) + 1), 1), max(len(z) - 1, 1))
    if len(z) < 20:
        return signal.sosfilt(sos, z.real) + 1j * signal.sosfilt(sos, z.imag)
    return signal.sosfiltfilt(sos, z.real, padlen=padlen) + 1j * signal.sosfiltfilt(sos, z.imag, padlen=padlen)


def preprocess_complex(z: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    mode = cfg.get("normalize", "robust_record")
    if mode == "robust_record":
        z = robust_normalize(z)
    elif mode == "zscore":
        scale = np.std(z)
        z = (z - np.mean(z)) / (scale if scale > 1e-12 else 1.0)
    elif mode not in {"none", None}:
        raise ValueError(f"Unknown normalization mode: {mode}")
    return highpass_complex(
        z,
        fs=fs,
        cutoff_hz=float(cfg["highpass_cutoff_hz"]),
        order=int(cfg["highpass_order"]),
        ripple_db=float(cfg.get("cheby_ripple_db", 0.5)),
    )


def autocorrelation_harmonic_ratio(x: np.ndarray, fs: float, min_f0: float = 25.0, max_f0: float = 500.0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    energy = float(np.dot(x, x))
    if len(x) < 8 or energy < 1e-12:
        return 0.0, float("nan")
    corr = signal.correlate(x, x, mode="full", method="fft")[len(x)-1:]
    denom = np.arange(len(x), 0, -1, dtype=float)
    corr = corr / np.maximum(denom, 1.0)
    corr = corr / max(corr[0], 1e-12)
    min_lag = max(int(fs / max_f0), 1)
    max_lag = min(int(fs / min_f0), len(corr) - 1)
    if max_lag <= min_lag:
        return 0.0, float("nan")
    segment_corr = corr[min_lag:max_lag + 1]
    peaks, _ = signal.find_peaks(segment_corr, prominence=0.02)
    if len(peaks) == 0:
        idx = int(np.argmax(segment_corr)) + min_lag
        hr = float(np.clip(corr[idx], 0.0, 1.0))
        return hr, float(fs / idx) if idx > 0 else float("nan")
    peak_lags = peaks + min_lag
    peak_values = corr[peak_lags]
    max_hr = float(np.max(peak_values))
    # Repeated harmonics can create comparable peaks at integer multiples of the
    # fundamental period. Select the earliest strong peak while retaining the
    # maximum peak value as the harmonicity measure.
    strong = peak_lags[peak_values >= 0.80 * max_hr]
    idx = int(np.min(strong)) if len(strong) else int(peak_lags[np.argmax(peak_values)])
    hr = float(np.clip(max_hr, 0.0, 1.0))
    f0 = float(fs / idx) if idx > 0 else float("nan")
    return hr, f0


def combined_psd(z: np.ndarray, fs: float, n_fft: int = 256) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(max(64, n_fft), len(z))
    if nperseg < 8:
        return np.asarray([0.0]), np.asarray([0.0])
    freqs, p_i = signal.welch(z.real, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2, nfft=max(n_fft, nperseg))
    _, p_q = signal.welch(z.imag, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2, nfft=max(n_fft, nperseg))
    return freqs, np.maximum(p_i + p_q, 1e-18)


def add_complex_awgn(z: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    power = float(np.mean(np.abs(z) ** 2))
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (rng.standard_normal(len(z)) + 1j * rng.standard_normal(len(z)))
    return z + noise
