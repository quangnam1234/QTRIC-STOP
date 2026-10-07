from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    logp = np.log(np.clip(probs, 1e-12, 1.0)) / max(float(temperature), 1e-6)
    logp -= np.max(logp, axis=1, keepdims=True)
    exp = np.exp(logp)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def fit_temperature(probs: np.ndarray, y_idx: np.ndarray, weights: np.ndarray | None, bounds=(0.25, 8.0)) -> float:
    probs = np.asarray(probs, dtype=float)
    y_idx = np.asarray(y_idx, dtype=int)
    if weights is None:
        weights = np.ones(len(y_idx), dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / max(weights.sum(), 1e-12)

    def objective(t: float) -> float:
        calibrated = _temperature_scale(probs, t)
        losses = -np.log(np.clip(calibrated[np.arange(len(y_idx)), y_idx], 1e-12, 1.0))
        return float(np.sum(weights * losses))

    result = minimize_scalar(objective, bounds=tuple(map(float, bounds)), method="bounded")
    return float(result.x if result.success else 1.0)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    return _temperature_scale(probs, temperature)
