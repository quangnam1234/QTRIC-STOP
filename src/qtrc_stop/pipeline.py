from __future__ import annotations

import json
import math
import os
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from .baselines import (
    apply_score_stopping, confidence_policy, fit_two_stage_reliability,
    fixed_duration_decisions, select_score_threshold,
)
from .calibration import apply_temperature, fit_temperature
from .checkpoint import RunState, atomic_dataframe_csv, atomic_write_json, task_lock
from .config import config_fingerprint, save_resolved_config
from .events import detect_events, events_to_frame
from .features import extract_event_feature_rows, feature_columns
from .io import ensure_dataset, inventory_dataset, read_signal_csv, select_inventory, write_inventory_reports
from .logging_utils import setup_logging
from .metrics import (
    brier_multiclass, decision_metrics, expected_calibration_error,
    global_classification_metrics, species_metrics,
)
from .models import HierarchicalClassifier, subject_balanced_weights
from .plots import plot_fixed_duration, plot_risk_earliness
from .robustness import perturb_signal
from .signal import complex_from_amplitude_phase, preprocess_complex
from .stopping import Policy, apply_policy, evaluate_policy_grid, select_policy, select_policy_from_grid
from .taxonomy import SPECIES, label_indices, taxonomic_loss_matrix


STAGES = ["cp0", "cp1", "cp2", "cp3", "cp4", "cp5", "cp6", "cp7", "cp8"]


class Pipeline:
    def __init__(self, cfg: dict[str, Any], run_dir: str | Path, dataset_override: str | None = None):
        self.cfg = cfg
        if dataset_override:
            self.cfg["dataset"]["path"] = str(Path(dataset_override).expanduser().resolve())
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state = RunState(self.run_dir)
        self.logger = setup_logging(self.run_dir, cfg["runtime"]["log_level"])
        fingerprint = config_fingerprint(self.cfg)
        meta_path = self.run_dir / "run_meta.json"
        if meta_path.exists():
            previous = json.loads(meta_path.read_text(encoding="utf-8"))
            if previous.get("config_fingerprint") != fingerprint:
                raise RuntimeError(
                    "The run directory contains checkpoints from a different configuration. "
                    "Use a new run directory or restore the original configuration."
                )
        save_resolved_config(self.cfg, self.run_dir / "resolved_config.yaml")
        atomic_write_json(meta_path, {"config_fingerprint": fingerprint, "pid": os.getpid()})

    def run(self, start_stage: str | None = None, end_stage: str | None = None):
        start = STAGES.index(start_stage) if start_stage else 0
        end = STAGES.index(end_stage) if end_stage else len(STAGES) - 1
        for stage in STAGES[start:end + 1]:
            method = getattr(self, stage)
            self.logger.info("=== %s START ===", stage.upper())
            self.state.update_stage(stage, "running")
            try:
                method()
                self.state.update_stage(stage, "done")
                self.logger.info("=== %s DONE ===", stage.upper())
            except Exception as exc:
                self.state.update_stage(stage, "failed", error=str(exc), traceback=traceback.format_exc())
                self.logger.exception("Stage %s failed", stage)
                raise

    @property
    def stale_minutes(self) -> int:
        return int(self.cfg["runtime"]["stale_lock_minutes"])

    def cp0(self):
        out = self.run_dir / "cp0_inventory"
        dataset_root = ensure_dataset(self.cfg["dataset"]["path"], self.run_dir / "data_cache" / "raw")
        df = inventory_dataset(dataset_root)
        write_inventory_reports(df, out / "all")
        expected = int(self.cfg["dataset"].get("expected_files", 0))
        if self.cfg["dataset"].get("strict_inventory", True) and expected and len(df) != expected:
            raise RuntimeError(f"Expected {expected} files, found {len(df)}")
        selected = select_inventory(
            df,
            self.cfg["dataset"].get("max_subjects_per_species"),
            self.cfg["dataset"].get("max_recordings_per_subject"),
        )
        write_inventory_reports(selected, out / "selected")
        self.logger.info("Selected %d recordings from %d subjects", len(selected), selected.subject_id.nunique())

    def cp1(self):
        inv = pd.read_csv(self.run_dir / "cp0_inventory" / "selected" / "recordings.csv")
        parts = self.run_dir / "cp1_events" / "parts"
        parts.mkdir(parents=True, exist_ok=True)
        for row in inv.itertuples(index=False):
            artifact = parts / f"{row.recording_id}.csv"
            task = f"cp1::{row.recording_id}"
            if self.state.task_done(task, artifact):
                continue
            with task_lock(artifact.with_suffix(".lock"), self.stale_minutes):
                amp, phase = read_signal_csv(row.path)
                z = complex_from_amplitude_phase(amp, phase)
                z_hp = preprocess_complex(z, float(self.cfg["dataset"]["sampling_rate_hz"]), self.cfg["signal"])
                events = detect_events(z_hp, float(self.cfg["dataset"]["sampling_rate_hz"]), self.cfg["events"])
                frame = events_to_frame(pd.Series(row._asdict()), events)
                atomic_dataframe_csv(frame, artifact)
                self.state.update_task(task, "done", events=len(frame), artifact=str(artifact))
        frames = [pd.read_csv(p) for p in sorted(parts.glob("*.csv")) if p.stat().st_size > 0]
        events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        atomic_dataframe_csv(events, self.run_dir / "cp1_events" / "events.csv")
        if events.empty:
            raise RuntimeError("No events detected")
        self.logger.info("Detected %d events; fallback fraction %.3f", len(events), events["fallback_event"].mean())

    def cp2(self):
        events = pd.read_csv(self.run_dir / "cp1_events" / "events.csv")
        parts = self.run_dir / "cp2_features" / "parts"
        parts.mkdir(parents=True, exist_ok=True)
        for recording_id, group in events.groupby("recording_id", sort=True):
            artifact = parts / f"{recording_id}.csv.gz"
            task = f"cp2::{recording_id}"
            if self.state.task_done(task, artifact):
                continue
            with task_lock(artifact.with_suffix(".lock"), self.stale_minutes):
                amp, phase = read_signal_csv(group.iloc[0]["path"])
                z = complex_from_amplitude_phase(amp, phase)
                z_hp = preprocess_complex(z, float(self.cfg["dataset"]["sampling_rate_hz"]), self.cfg["signal"])
                rows = []
                for _, event in group.iterrows():
                    rows.extend(extract_event_feature_rows(event, z_hp, float(self.cfg["dataset"]["sampling_rate_hz"]), self.cfg["features"]))
                frame = pd.DataFrame(rows)
                atomic_dataframe_csv(frame, artifact, compression="gzip")
                self.state.update_task(task, "done", rows=len(frame), artifact=str(artifact))
        frames = [pd.read_csv(p) for p in sorted(parts.glob("*.csv.gz"))]
        features = pd.concat(frames, ignore_index=True)
        artifact = self.run_dir / "cp2_features" / "features.csv.gz"
        atomic_dataframe_csv(features, artifact, compression="gzip")
        self.logger.info("Extracted %d prefix rows with %d numerical features", len(features), len(feature_columns(features)))

    def cp3(self):
        df = pd.read_csv(self.run_dir / "cp2_features" / "features.csv.gz")
        durations = list(map(float, self.cfg["features"]["prefix_durations_sec"]))
        report: dict[str, Any] = {}
        report["rows"] = int(len(df))
        report["events"] = int(df.event_id.nunique())
        report["subjects"] = int(df.subject_id.nunique())
        report["species"] = df.groupby("species")["subject_id"].nunique().astype(int).to_dict()
        report["nan_fraction"] = float(df[feature_columns(df)].isna().mean().mean())
        duration_counts = df.groupby("event_id")["duration_sec"].nunique()
        report["complete_event_fraction"] = float(np.mean(duration_counts == len(durations)))
        report["fallback_event_fraction"] = float(pd.read_csv(self.run_dir / "cp1_events" / "events.csv")["fallback_event"].mean())
        gates = self.cfg.get("gates", {})
        expected_species = gates.get("expected_subjects_per_species")
        expected_ok = True
        if expected_species:
            expected_ok = all(report["species"].get(k, 0) >= int(v) for k, v in expected_species.items())
        report["pass"] = bool(
            report["subjects"] >= int(gates.get("min_subjects_total", 10))
            and len(report["species"]) >= int(gates.get("min_species", 5))
            and report["nan_fraction"] <= float(gates.get("nan_fraction_max", 0.05))
            and report["complete_event_fraction"] >= float(gates.get("complete_event_fraction_min", 0.95))
            and report["fallback_event_fraction"] <= float(gates.get("fallback_event_fraction_max", 1.0))
            and expected_ok
        )
        report["gate_config"] = gates
        atomic_write_json(self.run_dir / "cp3_gates" / "gate_report.json", report)
        if not report["pass"] and self.cfg["runtime"].get("fail_on_gate", True):
            raise RuntimeError(f"CP3 gate failed: {report}")

    def _outer_splits(self, df: pd.DataFrame):
        subject_df = df[["subject_id", "species"]].drop_duplicates().sort_values("subject_id")
        X = np.zeros((len(subject_df), 1))
        y = subject_df["species"].to_numpy()
        groups = subject_df["subject_id"].to_numpy()
        protocol = self.cfg["cv"]["outer_protocol"]
        if protocol == "loso":
            splitter = LeaveOneGroupOut()
            splits = list(splitter.split(X, y, groups))
        else:
            n = int(self.cfg["cv"]["outer_splits"])
            splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=int(self.cfg["seed"]))
            splits = list(splitter.split(X, y, groups))
        max_folds = self.cfg["cv"].get("max_outer_folds")
        if max_folds is not None:
            splits = splits[: int(max_folds)]
        for fold, (tr, te) in enumerate(splits):
            yield fold, set(groups[tr]), set(groups[te])

    @staticmethod
    def _prepare_xy(df: pd.DataFrame, fcols: list[str]):
        x = df[fcols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        y = df["species"].to_numpy()
        w = subject_balanced_weights(df["subject_id"].to_numpy(), y)
        return x, y, w

    def _predict_task(self, train_df: pd.DataFrame, val_df: pd.DataFrame, duration: float, seed: int, artifact: Path, model_path: Path | None):
        fcols = feature_columns(train_df)
        x_train, y_train, w = self._prepare_xy(train_df, fcols)
        x_val, _, _ = self._prepare_xy(val_df, fcols)
        model = HierarchicalClassifier(self.cfg["model"], seed).fit(x_train, y_train, fcols, w)
        probs = model.predict_proba(x_val)
        pred = val_df[["event_id", "recording_id", "subject_id", "species", "duration_sec", "harmonic_ratio", "harmonic_energy_ratio", "f0_delta_hz"]].copy()
        for idx, species in enumerate(SPECIES):
            pred[f"p_{species}"] = probs[:, idx]
        atomic_dataframe_csv(pred, artifact, compression="gzip")
        if model_path is not None:
            model.save(model_path)

    def cp4(self):
        df = pd.read_csv(self.run_dir / "cp2_features" / "features.csv.gz")
        durations = list(map(float, self.cfg["features"]["prefix_durations_sec"]))
        root = self.run_dir / "cp4_predictions"
        for outer_fold, train_subjects, test_subjects in self._outer_splits(df):
            fold_dir = root / f"outer_{outer_fold:03d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            outer_train = df[df.subject_id.isin(train_subjects)].copy()
            outer_test = df[df.subject_id.isin(test_subjects)].copy()
            atomic_write_json(fold_dir / "split.json", {"train_subjects": sorted(train_subjects), "test_subjects": sorted(test_subjects)})

            subject_df = outer_train[["subject_id", "species"]].drop_duplicates().sort_values("subject_id")
            min_subjects = int(subject_df.groupby("species")["subject_id"].nunique().min())
            n_inner = max(2, min(int(self.cfg["cv"]["inner_splits"]), min_subjects))
            inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=int(self.cfg["seed"]) + outer_fold)
            Xs = np.zeros((len(subject_df), 1))
            ys = subject_df.species.to_numpy()
            gs = subject_df.subject_id.to_numpy()
            inner_parts = []
            for inner_fold, (itr, iva) in enumerate(inner.split(Xs, ys, gs)):
                inner_train_subjects = set(gs[itr])
                inner_val_subjects = set(gs[iva])
                for duration in durations:
                    artifact = fold_dir / "oof_parts" / f"inner_{inner_fold:02d}_t{duration:.1f}.csv.gz"
                    task = f"cp4::outer{outer_fold}::inner{inner_fold}::t{duration:.1f}"
                    if not self.state.task_done(task, artifact):
                        with task_lock(artifact.with_suffix(".lock"), self.stale_minutes):
                            tr = outer_train[outer_train.subject_id.isin(inner_train_subjects) & np.isclose(outer_train.duration_sec, duration)]
                            va = outer_train[outer_train.subject_id.isin(inner_val_subjects) & np.isclose(outer_train.duration_sec, duration)]
                            self._predict_task(tr, va, duration, int(self.cfg["seed"]) + outer_fold * 1000 + inner_fold * 100 + int(duration * 10), artifact, None)
                            self.state.update_task(task, "done", artifact=str(artifact))
                    inner_parts.append(artifact)
            oof = pd.concat([pd.read_csv(p) for p in sorted(set(inner_parts))], ignore_index=True)
            atomic_dataframe_csv(oof, fold_dir / "oof_predictions.csv.gz", compression="gzip")

            test_parts = []
            for duration in durations:
                artifact = fold_dir / "test_parts" / f"t{duration:.1f}.csv.gz"
                model_path = fold_dir / "models" / f"t{duration:.1f}.joblib"
                task = f"cp4::outer{outer_fold}::final::t{duration:.1f}"
                if not self.state.task_done(task, artifact):
                    with task_lock(artifact.with_suffix(".lock"), self.stale_minutes):
                        tr = outer_train[np.isclose(outer_train.duration_sec, duration)]
                        te = outer_test[np.isclose(outer_test.duration_sec, duration)]
                        self._predict_task(tr, te, duration, int(self.cfg["seed"]) + outer_fold * 1000 + 900 + int(duration * 10), artifact, model_path)
                        self.state.update_task(task, "done", artifact=str(artifact), model=str(model_path))
                test_parts.append(artifact)
            test = pd.concat([pd.read_csv(p) for p in test_parts], ignore_index=True)
            atomic_dataframe_csv(test, fold_dir / "test_predictions.csv.gz", compression="gzip")

    def _calibrate_fold(self, fold_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
        oof = pd.read_csv(fold_dir / "oof_predictions.csv.gz")
        test = pd.read_csv(fold_dir / "test_predictions.csv.gz")
        durations = list(map(float, self.cfg["features"]["prefix_durations_sec"]))
        temperatures: dict[str, float] = {}
        for duration in durations:
            mask = np.isclose(oof.duration_sec, duration)
            probs = oof.loc[mask, [f"p_{s}" for s in SPECIES]].to_numpy()
            y_idx = label_indices(oof.loc[mask, "species"])
            weights = subject_balanced_weights(oof.loc[mask, "subject_id"], oof.loc[mask, "species"])
            temp = fit_temperature(probs, y_idx, weights, self.cfg["calibration"]["temperature_bounds"])
            temperatures[f"{duration:.1f}"] = temp
            for frame in [oof, test]:
                m = np.isclose(frame.duration_sec, duration)
                p = frame.loc[m, [f"p_{s}" for s in SPECIES]].to_numpy()
                pc = apply_temperature(p, temp)
                for idx, species in enumerate(SPECIES):
                    frame.loc[m, f"p_{species}"] = pc[:, idx]
        return oof, test, temperatures

    def cp5(self):
        features = pd.read_csv(self.run_dir / "cp2_features" / "features.csv.gz")
        subject_counts = features[["subject_id", "species"]].drop_duplicates().groupby("species")["subject_id"].nunique().to_dict()
        loss_matrix = taxonomic_loss_matrix(
            subject_counts,
            float(self.cfg["policy"]["beta_family"]),
            float(self.cfg["policy"]["beta_genus"]),
            bool(self.cfg["policy"]["inverse_subject_class_weight"]),
        )
        atomic_write_json(self.run_dir / "cp5_policies" / "loss_matrix.json", {"species": SPECIES, "matrix": loss_matrix.tolist(), "subject_counts": subject_counts})
        durations = list(map(float, self.cfg["features"]["prefix_durations_sec"]))
        fold_dirs = sorted((self.run_dir / "cp4_predictions").glob("outer_*"))
        for fold_dir in fold_dirs:
            fold = int(fold_dir.name.split("_")[1])
            calibrated_oof = fold_dir / "calibrated_oof.csv.gz"
            calibrated_test = fold_dir / "calibrated_test.csv.gz"
            temp_path = fold_dir / "temperatures.json"
            task = f"cp5::outer{fold}::temperature"
            if not self.state.task_done(task, calibrated_test):
                oof, test, temps = self._calibrate_fold(fold_dir)
                atomic_dataframe_csv(oof, calibrated_oof, compression="gzip")
                atomic_dataframe_csv(test, calibrated_test, compression="gzip")
                atomic_write_json(temp_path, temps)
                self.state.update_task(task, "done", artifact=str(calibrated_test))
            else:
                oof = pd.read_csv(calibrated_oof)

            grid_path = fold_dir / "policy_grids" / "qtrc_grid.csv.gz"
            grid_task = f"cp5::outer{fold}::grid"
            if not self.state.task_done(grid_task, grid_path):
                with task_lock(grid_path.with_suffix(".lock"), self.stale_minutes):
                    grid = evaluate_policy_grid(oof, durations, loss_matrix, self.cfg["policy"], int(self.cfg["seed"]) + fold)
                    atomic_dataframe_csv(grid, grid_path, compression="gzip")
                    self.state.update_task(grid_task, "done", artifact=str(grid_path))
            else:
                grid = pd.read_csv(grid_path)

            for epsilon in map(float, self.cfg["policy"]["epsilons"]):
                policy_path = fold_dir / "policies" / f"epsilon_{epsilon:.3f}.json"
                task = f"cp5::outer{fold}::eps{epsilon:.3f}"
                if self.state.task_done(task, policy_path):
                    continue
                policy = select_policy_from_grid(grid, durations, epsilon, self.cfg["policy"])
                atomic_write_json(policy_path, asdict(policy))
                self.state.update_task(task, "done", artifact=str(policy_path))

    @staticmethod
    def _load_policy(path: Path) -> Policy:
        return Policy(**json.loads(path.read_text(encoding="utf-8")))

    def cp6(self):
        """Evaluate every method without changing any trained model or policy.

        Fold-level rows contain only metrics that are valid for a single held-out
        subject. Multiclass metrics are computed once from the pooled outer-test
        decisions across all LOSO folds.
        """
        loss_payload = json.loads((self.run_dir / "cp5_policies" / "loss_matrix.json").read_text(encoding="utf-8"))
        loss_matrix = np.asarray(loss_payload["matrix"], dtype=float)
        zero_one = np.ones((len(SPECIES), len(SPECIES)), dtype=float) - np.eye(len(SPECIES))
        durations = list(map(float, self.cfg["features"]["prefix_durations_sec"]))
        metric_rows, all_decision_parts, calibration_rows = [], [], []

        def record_decisions(frame: pd.DataFrame, fold: int, method: str, epsilon: float | None) -> pd.DataFrame:
            recorded = frame.copy()
            recorded["outer_fold"] = fold
            recorded["method"] = method
            recorded["epsilon"] = np.nan if epsilon is None else float(epsilon)
            all_decision_parts.append(recorded)
            return recorded

        for fold_dir in sorted((self.run_dir / "cp4_predictions").glob("outer_*")):
            fold = int(fold_dir.name.split("_")[1])
            raw_test = pd.read_csv(fold_dir / "test_predictions.csv.gz")
            test = pd.read_csv(fold_dir / "calibrated_test.csv.gz")
            oof = pd.read_csv(fold_dir / "calibrated_oof.csv.gz")

            for duration in durations:
                for status, frame in [("raw", raw_test), ("calibrated", test)]:
                    sub = frame[np.isclose(frame.duration_sec, duration)]
                    probs = sub[[f"p_{species}" for species in SPECIES]].to_numpy()
                    y_idx = label_indices(sub["species"])
                    calibration_rows.append({
                        "outer_fold": fold,
                        "duration_sec": duration,
                        "status": status,
                        "ece": expected_calibration_error(probs, y_idx),
                        "nll": float(log_loss(y_idx, probs, labels=list(range(len(SPECIES))))),
                        "brier": brier_multiclass(probs, y_idx),
                    })

                method = f"fixed_{duration:.1f}"
                decisions = fixed_duration_decisions(test, duration)
                metrics = decision_metrics(decisions, loss_matrix, max(durations))
                metric_rows.append({
                    "outer_fold": fold,
                    "method": method,
                    "epsilon": np.nan,
                    "risk_violation": np.nan,
                    **metrics,
                })
                record_decisions(decisions, fold, method, None)

            for status, frame in [("raw", raw_test), ("calibrated", test)]:
                for threshold in [0.70, 0.80, 0.90, 0.95]:
                    method = f"{status}_conf_{threshold:.2f}"
                    decisions = apply_policy(frame, durations, confidence_policy(threshold), selective_mode=False)
                    metrics = decision_metrics(decisions, loss_matrix, max(durations))
                    metric_rows.append({
                        "outer_fold": fold,
                        "method": method,
                        "epsilon": np.nan,
                        "risk_violation": np.nan,
                        **metrics,
                    })
                    record_decisions(decisions, fold, method, None)

            # TEASER-style two-stage reliability baseline. The implementation
            # is an internal reliability wrapper, not an exact reproduction.
            rel_oof, rel_test = fit_two_stage_reliability(
                oof, test, durations, int(self.cfg["seed"]) + 20000 + fold
            )

            # AAGC-style baseline: confidence-only policy calibrated under 0-1 gap loss.
            base_cfg = dict(self.cfg["policy"])
            base_cfg.update({
                "js_grid": [1.0],
                "harmonic_ratio_grid": [0.0],
                "harmonic_energy_grid": [0.0],
                "f0_delta_grid_hz": [1e9],
                "persistence_grid": [1],
            })
            base_grid = evaluate_policy_grid(
                oof,
                durations,
                zero_one,
                base_cfg,
                int(self.cfg["seed"]) + 10000 + fold,
            )

            for epsilon in map(float, self.cfg["policy"]["epsilons"]):
                rel_threshold, _rel_grid = select_score_threshold(
                    rel_oof,
                    durations,
                    epsilon,
                    zero_one,
                    self.cfg["policy"],
                    int(self.cfg["seed"]) + 21000 + fold,
                )
                rel_decisions = apply_score_stopping(rel_test, durations, rel_threshold)
                rel_metrics = decision_metrics(rel_decisions, loss_matrix, max(durations))
                metric_rows.append({
                    "outer_fold": fold,
                    "method": "two_stage_reliability",
                    "epsilon": epsilon,
                    "risk_violation": float(rel_metrics["subject_mean_gap"] > epsilon),
                    **rel_metrics,
                })
                record_decisions(rel_decisions, fold, "two_stage_reliability", epsilon)

                base_policy = select_policy_from_grid(base_grid, durations, epsilon, base_cfg)
                base_decisions = apply_policy(test, durations, base_policy, selective_mode=False)
                base_metrics = decision_metrics(base_decisions, loss_matrix, max(durations))
                metric_rows.append({
                    "outer_fold": fold,
                    "method": "aagc_style",
                    "epsilon": epsilon,
                    "calibration_ucb_gap": base_policy.calibration_ucb_gap,
                    "risk_violation": float(base_metrics["subject_mean_gap"] > epsilon),
                    **base_metrics,
                })
                record_decisions(base_decisions, fold, "aagc_style", epsilon)

                policy = self._load_policy(fold_dir / "policies" / f"epsilon_{epsilon:.3f}.json")
                decisions = apply_policy(
                    test,
                    durations,
                    policy,
                    selective_mode=bool(self.cfg["policy"]["selective_mode"]),
                )
                metrics = decision_metrics(decisions, loss_matrix, max(durations))
                metric_rows.append({
                    "outer_fold": fold,
                    "method": "qtrc_stop",
                    "epsilon": epsilon,
                    "calibration_ucb_gap": policy.calibration_ucb_gap,
                    "risk_violation": float(metrics["subject_mean_gap"] > epsilon),
                    **metrics,
                })
                record_decisions(decisions, fold, "qtrc_stop", epsilon)

        metrics_df = pd.DataFrame(metric_rows)
        all_decisions_df = pd.concat(all_decision_parts, ignore_index=True) if all_decision_parts else pd.DataFrame()
        calibration_df = pd.DataFrame(calibration_rows)

        summary_rows = []
        for (method, epsilon), fold_group in metrics_df.groupby(["method", "epsilon"], dropna=False, sort=True):
            if pd.isna(epsilon):
                pooled = all_decisions_df[
                    (all_decisions_df["method"] == method) & all_decisions_df["epsilon"].isna()
                ]
            else:
                pooled = all_decisions_df[
                    (all_decisions_df["method"] == method)
                    & np.isclose(all_decisions_df["epsilon"].astype(float), float(epsilon), equal_nan=False)
                ]

            class_metrics = global_classification_metrics(pooled)
            summary_rows.append({
                "method": method,
                "epsilon": epsilon,
                "folds": int(fold_group["outer_fold"].nunique()),
                # Subject-macro accuracy is retained for backward compatibility.
                "accuracy_mean": float(fold_group["accuracy"].mean()),
                "accuracy_event_global": class_metrics["accuracy_event_global"],
                # These three values are pooled across all outer-test folds.
                "macro_f1_mean": class_metrics["macro_f1"],
                "balanced_accuracy_mean": class_metrics["balanced_accuracy"],
                "worst_recall_mean": class_metrics["worst_recall"],
                "mean_stop_sec": float(fold_group["mean_stop_sec"].mean()),
                "p90_stop_sec": float(fold_group["p90_stop_sec"].mean()),
                "earliness_mean": float(fold_group["earliness"].mean()),
                "subject_mean_gap": float(fold_group["subject_mean_gap"].mean()),
                "risk_violation_rate": float(fold_group["risk_violation"].mean()),
                "class_metric_scope": "pooled_outer_test_events",
            })
        summary = pd.DataFrame(summary_rows)

        qtrc_decisions = all_decisions_df[all_decisions_df["method"] == "qtrc_stop"].copy()
        species_rows = []
        for epsilon in map(float, self.cfg["policy"]["epsilons"]):
            pooled = qtrc_decisions[
                np.isclose(qtrc_decisions["epsilon"].astype(float), epsilon, equal_nan=False)
            ]
            sm = species_metrics(pooled)
            sm["method"] = "qtrc_stop"
            sm["epsilon"] = epsilon
            species_rows.append(sm)
        species_df = pd.concat(species_rows, ignore_index=True) if species_rows else pd.DataFrame()

        out = self.run_dir / "cp6_results"
        atomic_dataframe_csv(metrics_df, out / "fold_metrics.csv")
        atomic_dataframe_csv(species_df, out / "species_metrics.csv")
        atomic_dataframe_csv(qtrc_decisions, out / "event_decisions.csv.gz", compression="gzip")
        atomic_dataframe_csv(all_decisions_df, out / "all_method_event_decisions.csv.gz", compression="gzip")
        atomic_dataframe_csv(calibration_df, out / "calibration_metrics.csv")
        atomic_dataframe_csv(summary, out / "summary_metrics.csv")

    def cp7(self):
        out = self.run_dir / "cp7_robustness"
        enabled = bool(self.cfg.get("robustness", {}).get("enabled", False))
        if not enabled:
            atomic_write_json(out / "status.json", {"enabled": False, "status": "skipped"})
            return

        import hashlib

        loss_payload = json.loads((self.run_dir / "cp5_policies" / "loss_matrix.json").read_text(encoding="utf-8"))
        loss_matrix = np.asarray(loss_payload["matrix"], dtype=float)
        durations = list(map(float, self.cfg["features"]["prefix_durations_sec"]))
        all_events = pd.read_csv(self.run_dir / "cp1_events" / "events.csv")
        conditions = []
        for snr in self.cfg["robustness"].get("snr_db", []):
            conditions.append({"name": f"snr_{snr}dB", "snr_db": snr})
        for std in self.cfg["robustness"].get("phase_noise_std_rad", []):
            conditions.append({"name": f"phase_noise_{std}", "phase_noise_std_rad": std})
        for scale in self.cfg["robustness"].get("amplitude_scales", []):
            conditions.append({"name": f"amplitude_{scale}", "amplitude_scale": scale})
        atomic_write_json(out / "conditions.json", conditions)

        metric_rows, decision_parts = [], []
        for fold_dir in sorted((self.run_dir / "cp4_predictions").glob("outer_*")):
            fold = int(fold_dir.name.split("_")[1])
            split = json.loads((fold_dir / "split.json").read_text(encoding="utf-8"))
            test_subjects = set(split["test_subjects"])
            fold_events = all_events[all_events.subject_id.isin(test_subjects)].copy()
            temperatures = json.loads((fold_dir / "temperatures.json").read_text(encoding="utf-8"))

            for condition in conditions:
                name = str(condition["name"]).replace("/", "_")
                pred_path = out / f"outer_{fold:03d}" / name / "predictions.csv.gz"
                task = f"cp7::outer{fold}::{name}"
                if not self.state.task_done(task, pred_path):
                    with task_lock(pred_path.with_suffix(".lock"), self.stale_minutes):
                        feature_parts = []
                        for recording_id, group in fold_events.groupby("recording_id", sort=True):
                            amp, phase = read_signal_csv(group.iloc[0]["path"])
                            z = complex_from_amplitude_phase(amp, phase)
                            key = f"{self.cfg['seed']}::{fold}::{name}::{recording_id}".encode("utf-8")
                            cond_seed = int(hashlib.sha256(key).hexdigest()[:8], 16)
                            z = perturb_signal(z, condition, cond_seed)
                            z_hp = preprocess_complex(
                                z,
                                float(self.cfg["dataset"]["sampling_rate_hz"]),
                                self.cfg["signal"],
                            )
                            rows = []
                            for _, event in group.iterrows():
                                rows.extend(
                                    extract_event_feature_rows(
                                        event,
                                        z_hp,
                                        float(self.cfg["dataset"]["sampling_rate_hz"]),
                                        self.cfg["features"],
                                    )
                                )
                            feature_parts.append(pd.DataFrame(rows))
                        degraded = pd.concat(feature_parts, ignore_index=True)
                        pred_parts = []
                        for duration in durations:
                            sub = degraded[np.isclose(degraded.duration_sec, duration)].copy()
                            model = HierarchicalClassifier.load(fold_dir / "models" / f"t{duration:.1f}.joblib")
                            x = sub[model.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
                            probs = model.predict_proba(x)
                            probs = apply_temperature(probs, float(temperatures[f"{duration:.1f}"]))
                            pred = sub[[
                                "event_id", "recording_id", "subject_id", "species", "duration_sec",
                                "harmonic_ratio", "harmonic_energy_ratio", "f0_delta_hz",
                            ]].copy()
                            for k, species in enumerate(SPECIES):
                                pred[f"p_{species}"] = probs[:, k]
                            pred_parts.append(pred)
                        pred_df = pd.concat(pred_parts, ignore_index=True)
                        atomic_dataframe_csv(pred_df, pred_path, compression="gzip")
                        self.state.update_task(task, "done", artifact=str(pred_path))
                else:
                    pred_df = pd.read_csv(pred_path)

                for epsilon in map(float, self.cfg["policy"]["epsilons"]):
                    policy = self._load_policy(fold_dir / "policies" / f"epsilon_{epsilon:.3f}.json")
                    decisions = apply_policy(
                        pred_df,
                        durations,
                        policy,
                        selective_mode=bool(self.cfg["policy"]["selective_mode"]),
                    )
                    metrics = decision_metrics(decisions, loss_matrix, max(durations))
                    metric_rows.append({
                        "outer_fold": fold,
                        "condition": name,
                        "epsilon": epsilon,
                        **metrics,
                    })
                    recorded = decisions.copy()
                    recorded["outer_fold"] = fold
                    recorded["condition"] = name
                    recorded["epsilon"] = epsilon
                    decision_parts.append(recorded)

        metrics_df = pd.DataFrame(metric_rows)
        decisions_df = pd.concat(decision_parts, ignore_index=True) if decision_parts else pd.DataFrame()
        atomic_dataframe_csv(metrics_df, out / "robustness_metrics.csv")
        atomic_dataframe_csv(decisions_df, out / "robustness_event_decisions.csv.gz", compression="gzip")

        if not metrics_df.empty:
            summary_rows = []
            for (condition, epsilon), fold_group in metrics_df.groupby(["condition", "epsilon"], sort=True):
                pooled = decisions_df[
                    (decisions_df["condition"] == condition)
                    & np.isclose(decisions_df["epsilon"].astype(float), float(epsilon), equal_nan=False)
                ]
                class_metrics = global_classification_metrics(pooled)
                summary_rows.append({
                    "condition": condition,
                    "epsilon": epsilon,
                    "accuracy": float(fold_group["accuracy"].mean()),
                    "accuracy_event_global": class_metrics["accuracy_event_global"],
                    "macro_f1": class_metrics["macro_f1"],
                    "balanced_accuracy": class_metrics["balanced_accuracy"],
                    "worst_recall": class_metrics["worst_recall"],
                    "mean_stop_sec": float(fold_group["mean_stop_sec"].mean()),
                    "subject_mean_gap": float(fold_group["subject_mean_gap"].mean()),
                    "class_metric_scope": "pooled_outer_test_events",
                })
            atomic_dataframe_csv(pd.DataFrame(summary_rows), out / "robustness_summary.csv")

        atomic_write_json(out / "status.json", {
            "enabled": True,
            "status": "complete",
            "conditions": len(conditions),
            "note": "Synthetic perturbations are sensitivity analyses and do not establish field-domain robustness.",
        })

    def cp8(self):
        summary = pd.read_csv(self.run_dir / "cp6_results" / "summary_metrics.csv")
        fig_dir = self.run_dir / "cp8_reports" / "figures"
        plot_fixed_duration(summary, fig_dir / "fixed_duration_curve.png")
        plot_risk_earliness(summary, fig_dir / "risk_earliness.png")
        report = {
            "status": "complete",
            "config_fingerprint": config_fingerprint(self.cfg),
            "rows_in_summary": int(len(summary)),
            "required_outputs": [
                "cp0_inventory/selected/inventory.json",
                "cp1_events/events.csv",
                "cp2_features/features.csv.gz",
                "cp3_gates/gate_report.json",
                "cp4_predictions/outer_*/test_predictions.csv.gz",
                "cp5_policies/loss_matrix.json",
                "cp6_results/summary_metrics.csv",
                "cp8_reports/figures/fixed_duration_curve.png",
            ],
        }
        atomic_write_json(self.run_dir / "cp8_reports" / "final_report.json", report)
