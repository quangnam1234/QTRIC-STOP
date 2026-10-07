from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from .taxonomy import SPECIES, label_indices


CLASS_METRIC_KEYS = ("macro_f1", "balanced_accuracy", "worst_recall")


def expected_calibration_error(probs: np.ndarray, y_idx: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_idx).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if np.any(mask):
            ece += np.mean(mask) * abs(np.mean(correct[mask]) - np.mean(conf[mask]))
    return float(ece)


def brier_multiclass(probs: np.ndarray, y_idx: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[y_idx]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def decision_metrics(decisions: pd.DataFrame, loss_matrix: np.ndarray, max_duration: float) -> dict[str, float]:
    """Return metrics that are valid for one outer LOSO fold.

    One outer LOSO fold contains one held-out subject and therefore normally only
    one true species. Multiclass metrics such as macro-F1, balanced accuracy and
    worst-class recall are not identifiable from that fold alone. Their fields
    are deliberately returned as NaN and must be computed after pooling the
    outer-test decisions from every fold with :func:`global_classification_metrics`.

    This function does not change predictions, stopping decisions or any model
    algorithm. It only prevents absent species from being counted as zero-recall
    classes inside a single-subject fold.
    """
    if max_duration <= 0:
        raise ValueError("max_duration must be positive")

    active = decisions[~decisions["abstain"]].copy()
    coverage = len(active) / max(len(decisions), 1)
    mean_stop = float(decisions["stop_duration_sec"].mean())
    base = {
        "coverage": float(coverage),
        "mean_stop_sec": mean_stop,
        "median_stop_sec": float(decisions["stop_duration_sec"].median()),
        "p90_stop_sec": float(decisions["stop_duration_sec"].quantile(0.90)),
        "earliness": float(1.0 - mean_stop / max_duration),
    }
    empty_class_metrics = {key: np.nan for key in CLASS_METRIC_KEYS}

    if active.empty:
        return {
            **base,
            "accuracy": np.nan,
            **empty_class_metrics,
            "mean_taxonomic_loss": np.nan,
            "subject_mean_gap": np.nan,
            "subject_max_gap": np.nan,
        }

    y_true = active["species"].to_numpy()
    y_pred = active["pred_species"].to_numpy()
    true_idx = label_indices(y_true)
    pred_idx = label_indices(y_pred)
    full_idx = label_indices(active["full_pred_species"].to_numpy())
    losses = loss_matrix[true_idx, pred_idx]
    gaps = np.maximum(losses - loss_matrix[true_idx, full_idx], 0.0)

    tmp = active[["subject_id"]].copy()
    tmp["gap"] = gaps
    subject_gap = tmp.groupby("subject_id", sort=False)["gap"].mean().to_numpy()

    return {
        **base,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        **empty_class_metrics,
        "mean_taxonomic_loss": float(np.mean(losses)),
        "subject_mean_gap": float(np.mean(subject_gap)),
        "subject_max_gap": float(np.max(subject_gap)),
    }


def global_classification_metrics(decisions: pd.DataFrame) -> dict[str, float]:
    """Compute multiclass metrics after pooling all outer-test folds.

    The pooled table must contain every species in ``SPECIES`` among the true
    labels. This explicit check prevents an incomplete set of outer folds from
    silently producing a misleading zero for an absent class.
    """
    present = set(map(str, decisions["species"].dropna().unique()))
    missing = [species for species in SPECIES if species not in present]
    if missing:
        raise ValueError(
            "Global multiclass metrics require pooled outer-test decisions "
            f"containing every species. Missing true species: {missing}"
        )

    active = decisions[~decisions["abstain"]].copy()
    if active.empty:
        return {
            "accuracy_event_global": np.nan,
            "macro_f1": np.nan,
            "balanced_accuracy": np.nan,
            "worst_recall": np.nan,
        }

    y_true = active["species"].to_numpy()
    y_pred = active["pred_species"].to_numpy()
    recalls = recall_score(
        y_true,
        y_pred,
        labels=SPECIES,
        average=None,
        zero_division=0,
    )
    return {
        "accuracy_event_global": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=SPECIES,
                average="macro",
                zero_division=0,
            )
        ),
        # With all classes present, balanced accuracy is the unweighted mean
        # of per-class recalls. Computing it directly also fixes the label order.
        "balanced_accuracy": float(np.mean(recalls)),
        "worst_recall": float(np.min(recalls)),
    }


def species_metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    """Return per-species metrics from pooled outer-test decisions."""
    rows = []
    for species in SPECIES:
        sub = decisions[decisions["species"] == species]
        if sub.empty:
            continue
        active = sub[~sub["abstain"]]
        recall_covered = float(np.mean(active["pred_species"] == species)) if len(active) else 0.0
        recall_total = float(np.mean((sub["pred_species"] == species) & (~sub["abstain"])))
        rows.append(
            {
                "species": species,
                "n_subjects": int(sub["subject_id"].nunique()),
                "n_events": int(len(sub)),
                "coverage": float(len(active) / len(sub)),
                "recall_covered": recall_covered,
                "recall_total": recall_total,
                "fnr_total": float(1.0 - recall_total),
                "mean_stop_sec": float(sub["stop_duration_sec"].mean()),
            }
        )
    return pd.DataFrame(rows)
