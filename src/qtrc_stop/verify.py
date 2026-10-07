from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REQUIRED = [
    "cp0_inventory/selected/inventory.json",
    "cp1_events/events.csv",
    "cp2_features/features.csv.gz",
    "cp3_gates/gate_report.json",
    "cp5_policies/loss_matrix.json",
    "cp6_results/fold_metrics.csv",
    "cp6_results/summary_metrics.csv",
    "cp8_reports/final_report.json",
]


def verify_run(run_dir: str | Path) -> tuple[bool, dict]:
    root = Path(run_dir)
    report = {"run_dir": str(root.resolve()), "missing": [], "failed_stages": [], "warnings": []}
    for rel in REQUIRED:
        if not (root / rel).exists():
            report["missing"].append(rel)
    state_path = root / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        report["failed_stages"] = [k for k, v in state.get("stages", {}).items() if v.get("status") == "failed"]
        report["stages"] = {k: v.get("status") for k, v in state.get("stages", {}).items()}
    else:
        report["missing"].append("state.json")
    gate_path = root / "cp3_gates" / "gate_report.json"
    if gate_path.exists():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        report["cp3_pass"] = bool(gate.get("pass", False))
        if not report["cp3_pass"]:
            report["warnings"].append("CP3 gate is not PASS")
    summary_path = root / "cp6_results" / "summary_metrics.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        report["summary_rows"] = int(len(summary))
        report["methods"] = sorted(summary["method"].dropna().unique().tolist())
        if summary.empty:
            report["warnings"].append("Summary table is empty")
    ok = not report["missing"] and not report["failed_stages"] and report.get("cp3_pass", False) and report.get("summary_rows", 0) > 0
    report["ok"] = bool(ok)
    return bool(ok), report
