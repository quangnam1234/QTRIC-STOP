import numpy as np
import pandas as pd
import pytest

from qtrc_stop.metrics import decision_metrics, global_classification_metrics
from qtrc_stop.taxonomy import SPECIES


def _decision_row(subject_id: str, species: str, pred_species: str) -> dict:
    return {
        "subject_id": subject_id,
        "species": species,
        "pred_species": pred_species,
        "full_pred_species": species,
        "abstain": False,
        "stop_duration_sec": 1.0,
    }


def test_single_subject_fold_does_not_report_multiclass_metrics():
    decisions = pd.DataFrame([
        _decision_row("s1", "apis_mellifera", "apis_mellifera"),
        _decision_row("s1", "apis_mellifera", "bombus_lapidarius"),
    ])
    loss_matrix = np.ones((len(SPECIES), len(SPECIES))) - np.eye(len(SPECIES))

    metrics = decision_metrics(decisions, loss_matrix, max_duration=2.0)

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert np.isnan(metrics["macro_f1"])
    assert np.isnan(metrics["balanced_accuracy"])
    assert np.isnan(metrics["worst_recall"])


def test_global_metrics_are_computed_after_pooling_all_species():
    rows = []
    for i, species in enumerate(SPECIES):
        rows.append(_decision_row(f"{species}_s1", species, species))
        rows.append(_decision_row(f"{species}_s2", species, species))

    # One of the two bombus_terrestris events is misclassified as apis_mellifera.
    rows[2 * SPECIES.index("bombus_terrestris")]["pred_species"] = "apis_mellifera"
    decisions = pd.DataFrame(rows)

    metrics = global_classification_metrics(decisions)

    assert metrics["accuracy_event_global"] == pytest.approx(0.9)
    assert metrics["balanced_accuracy"] == pytest.approx(0.9)
    assert metrics["worst_recall"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx((0.8 + 1.0 + 1.0 + 2.0 / 3.0 + 1.0) / 5.0)


def test_global_metrics_reject_incomplete_outer_fold_collection():
    decisions = pd.DataFrame([
        _decision_row("s1", species, species)
        for species in SPECIES[:-1]
    ])

    with pytest.raises(ValueError, match="Missing true species"):
        global_classification_metrics(decisions)
