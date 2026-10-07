from __future__ import annotations

import numpy as np

from .signal import add_complex_awgn


def perturb_signal(z: np.ndarray, condition: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(z, dtype=np.complex128).copy()
    if "amplitude_scale" in condition:
        out *= float(condition["amplitude_scale"])
    if "phase_noise_std_rad" in condition:
        phase_noise = rng.normal(0.0, float(condition["phase_noise_std_rad"]), size=len(out))
        out *= np.exp(1j * phase_noise)
    if "snr_db" in condition:
        out = add_complex_awgn(out, float(condition["snr_db"]), rng)
    return out
