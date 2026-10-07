from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from .taxonomy import SPECIES, label_indices


@dataclass(frozen=True)
class Policy:
    confidence: float
    js_max: float
    harmonic_ratio_min: float
    harmonic_energy_min: float
    f0_delta_max_hz: float
    persistence: int
    epsilon: float
    bound_method: str
    calibration_mean_gap: float
    calibration_ucb_gap: float
    calibration_mean_time: float


def _pivot_probabilities(pred_df: pd.DataFrame, durations: list[float]) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    events = sorted(pred_df["event_id"].unique())
    meta = pred_df.drop_duplicates("event_id").set_index("event_id").loc[events]
    probs = np.zeros((len(events), len(durations), len(SPECIES)), dtype=float)
    for j, duration in enumerate(durations):
        sub = pred_df[np.isclose(pred_df["duration_sec"], duration)].set_index("event_id").reindex(events)
        if sub[[f"p_{s}" for s in SPECIES]].isna().any().any():
            raise ValueError(f"Missing prefix predictions at duration {duration}")
        for k, species in enumerate(SPECIES):
            probs[:, j, k] = sub[f"p_{species}"].to_numpy()
    return events, probs, meta


def _quality_arrays(pred_df: pd.DataFrame, events: list[str], durations: list[float]) -> dict[str, np.ndarray]:
    out = {k: np.zeros((len(events), len(durations)), dtype=float) for k in ["harmonic_ratio", "harmonic_energy_ratio", "f0_delta_hz"]}
    for j, duration in enumerate(durations):
        sub = pred_df[np.isclose(pred_df["duration_sec"], duration)].set_index("event_id").reindex(events)
        for k in out:
            out[k][:, j] = sub[k].fillna(0.0).to_numpy()
    return out


def _js_array(probs: np.ndarray) -> np.ndarray:
    n, t, _ = probs.shape
    js = np.zeros((n, t), dtype=float)
    for j in range(1, t):
        js[:, j] = np.asarray([jensenshannon(probs[i, j], probs[i, j - 1], base=2.0) ** 2 for i in range(n)])
    return js


def prepare_policy_data(pred_df: pd.DataFrame, durations: list[float]) -> dict[str, Any]:
    events, probs, meta = _pivot_probabilities(pred_df, durations)
    quality = _quality_arrays(pred_df, events, durations)
    labels = probs.argmax(axis=2)
    persistence = {
        1: np.ones(labels.shape, dtype=bool),
        2: np.concatenate([np.zeros((len(events), 1), dtype=bool), labels[:, 1:] == labels[:, :-1]], axis=1),
    }
    return {
        "events": events,
        "probs": probs,
        "meta": meta,
        "labels": labels,
        "confidence": probs.max(axis=2),
        "js": _js_array(probs),
        "quality": quality,
        "persistence": persistence,
        "durations": np.asarray(durations, dtype=float),
        "true_idx": label_indices(meta["species"].to_numpy()),
        "subject_ids": meta["subject_id"].to_numpy(),
    }


def _candidate_indices(data: dict[str, Any], policy: Policy) -> tuple[np.ndarray, np.ndarray]:
    p_mask = data["persistence"].get(policy.persistence)
    if p_mask is None:
        labels = data["labels"]
        p_mask = np.zeros(labels.shape, dtype=bool)
        for j in range(policy.persistence - 1, labels.shape[1]):
            window = labels[:, j - policy.persistence + 1:j + 1]
            p_mask[:, j] = np.all(window == window[:, [-1]], axis=1)
    eligible = (
        (data["confidence"] >= policy.confidence)
        & (data["js"] <= policy.js_max)
        & (data["quality"]["harmonic_ratio"] >= policy.harmonic_ratio_min)
        & (data["quality"]["harmonic_energy_ratio"] >= policy.harmonic_energy_min)
        & (data["quality"]["f0_delta_hz"] <= policy.f0_delta_max_hz)
        & p_mask
    )
    has = eligible.any(axis=1)
    idx = np.where(has, eligible.argmax(axis=1), eligible.shape[1] - 1)
    return idx.astype(int), has


def _decisions_from_data(data: dict[str, Any], policy: Policy, selective_mode: bool) -> pd.DataFrame:
    idx, has = _candidate_indices(data, policy)
    row = np.arange(len(idx))
    pred_idx = data["labels"][row, idx]
    full_idx = data["labels"][:, -1]
    abstain = ~has if selective_mode else np.zeros(len(idx), dtype=bool)
    meta = data["meta"]
    return pd.DataFrame(
        {
            "event_id": data["events"],
            "subject_id": meta["subject_id"].to_numpy(),
            "species": meta["species"].to_numpy(),
            "pred_species": [SPECIES[i] for i in pred_idx],
            "stop_duration_sec": data["durations"][idx],
            "confidence": data["confidence"][row, idx],
            "abstain": abstain,
            "full_pred_species": [SPECIES[i] for i in full_idx],
        }
    )


def apply_policy(pred_df: pd.DataFrame, durations: list[float], policy: Policy, selective_mode: bool = False) -> pd.DataFrame:
    return _decisions_from_data(prepare_policy_data(pred_df, durations), policy, selective_mode)


def _subject_gap_from_arrays(data: dict[str, Any], idx: np.ndarray, loss_matrix: np.ndarray) -> np.ndarray:
    row = np.arange(len(idx))
    early_idx = data["labels"][row, idx]
    full_idx = data["labels"][:, -1]
    true_idx = data["true_idx"]
    gaps = np.maximum(loss_matrix[true_idx, early_idx] - loss_matrix[true_idx, full_idx], 0.0)
    subject_ids, inv = np.unique(data["subject_ids"], return_inverse=True)
    sums = np.bincount(inv, weights=gaps, minlength=len(subject_ids))
    counts = np.bincount(inv, minlength=len(subject_ids))
    return sums / np.maximum(counts, 1)


def risk_ucb(subject_losses: np.ndarray, delta: float, method: str, repeats: int, seed: int) -> float:
    x = np.asarray(subject_losses, dtype=float)
    if len(x) == 0:
        return 1.0
    mean = float(np.mean(x))
    if method == "empirical":
        return mean
    if method == "hoeffding_subject":
        return min(1.0, mean + np.sqrt(np.log(1.0 / delta) / (2.0 * len(x))))
    if method == "empirical_bernstein_subject":
        var = float(np.var(x, ddof=1)) if len(x) > 1 else 0.0
        term = np.sqrt(2.0 * var * np.log(3.0 / delta) / max(len(x), 1)) + 3.0 * np.log(3.0 / delta) / max(len(x), 1)
        return min(1.0, mean + term)
    if method == "bootstrap_subject":
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(x), size=(int(repeats), len(x)))
        means = x[indices].mean(axis=1)
        return float(np.quantile(means, 1.0 - delta))
    raise ValueError(f"Unknown bound method: {method}")


def policy_grid(cfg: dict) -> list[tuple[float, float, float, float, float, int]]:
    return list(
        itertools.product(
            cfg["confidence_grid"], cfg["js_grid"], cfg["harmonic_ratio_grid"],
            cfg["harmonic_energy_grid"], cfg["f0_delta_grid_hz"], cfg["persistence_grid"],
        )
    )


def evaluate_policy_grid(pred_df: pd.DataFrame, durations: list[float], loss_matrix: np.ndarray, cfg: dict, seed: int) -> pd.DataFrame:
    data = prepare_policy_data(pred_df, durations)
    records: list[dict[str, Any]] = []
    for conf, js_max, hr_min, he_min, f0_delta, persistence in policy_grid(cfg):
        candidate = Policy(float(conf), float(js_max), float(hr_min), float(he_min), float(f0_delta), int(persistence), 0.0, str(cfg["bound_method"]), 0.0, 0.0, 0.0)
        idx, _ = _candidate_indices(data, candidate)
        subject_losses = _subject_gap_from_arrays(data, idx, loss_matrix)
        records.append(
            {
                "confidence": float(conf), "js_max": float(js_max),
                "harmonic_ratio_min": float(hr_min), "harmonic_energy_min": float(he_min),
                "f0_delta_max_hz": float(f0_delta), "persistence": int(persistence),
                "mean_gap": float(np.mean(subject_losses)),
                "ucb_gap": risk_ucb(subject_losses, float(cfg["delta"]), str(cfg["bound_method"]), int(cfg["bootstrap_repeats"]), seed),
                "mean_time": float(np.mean(data["durations"][idx])),
            }
        )
    return pd.DataFrame(records)


def select_policy_from_grid(grid_df: pd.DataFrame, durations: list[float], epsilon: float, cfg: dict) -> Policy:
    feasible = grid_df[grid_df["ucb_gap"] <= float(epsilon)].copy()
    if feasible.empty:
        return Policy(1.1, 0.0, 1.1, 1.1, 0.0, 99, float(epsilon), str(cfg["bound_method"]), 0.0, 0.0, float(max(durations)))
    row = feasible.sort_values(["mean_time", "ucb_gap", "confidence"], ascending=[True, True, False]).iloc[0]
    return Policy(
        float(row.confidence), float(row.js_max), float(row.harmonic_ratio_min),
        float(row.harmonic_energy_min), float(row.f0_delta_max_hz), int(row.persistence),
        float(epsilon), str(cfg["bound_method"]), float(row.mean_gap), float(row.ucb_gap), float(row.mean_time),
    )


def select_policy(pred_df: pd.DataFrame, durations: list[float], epsilon: float, loss_matrix: np.ndarray, cfg: dict, seed: int) -> tuple[Policy, pd.DataFrame]:
    grid = evaluate_policy_grid(pred_df, durations, loss_matrix, cfg, seed)
    return select_policy_from_grid(grid, durations, epsilon, cfg), grid
