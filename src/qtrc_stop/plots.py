from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_fixed_duration(summary: pd.DataFrame, out_path: Path) -> None:
    """Plot fixed-duration results from the globally aggregated CP6 summary."""
    sub = summary[summary["method"].str.startswith("fixed_")].copy()
    if sub.empty:
        return
    sub["duration"] = sub["method"].str.replace("fixed_", "", regex=False).astype(float)
    sub = sub.sort_values("duration")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(sub["duration"], sub["accuracy_mean"], marker="o", label="Subject-macro accuracy")
    ax.plot(sub["duration"], sub["macro_f1_mean"], marker="s", label="Global Macro-F1")
    ax.set_xlabel("Observation duration (s)")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_risk_earliness(summary: pd.DataFrame, out_path: Path) -> None:
    """Plot QTRC-Stop risk/earliness points using global Macro-F1."""
    sub = summary[summary["method"] == "qtrc_stop"].copy().sort_values("epsilon")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(sub["mean_stop_sec"], sub["macro_f1_mean"], marker="o")
    for _, row in sub.iterrows():
        ax.annotate(f"eps={row.epsilon:.2f}", (row.mean_stop_sec, row.macro_f1_mean))
    ax.set_xlabel("Mean observation duration (s)")
    ax.set_ylabel("Global Macro-F1")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
