from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signal import autocorrelation_harmonic_ratio


EVENT_COLUMNS = [
    "event_id", "recording_id", "subject_id", "species", "path",
    "start_sample", "detector_end_sample", "detector_harmonic_ratio",
    "detector_f0_hz", "fallback_event",
]


@dataclass(frozen=True)
class EventCandidate:
    start_sample: int
    detector_end_sample: int
    harmonic_ratio: float
    detector_f0_hz: float
    fallback: bool = False


def detect_events(z_hp: np.ndarray, fs: float, cfg: dict) -> list[EventCandidate]:
    window = max(int(round(float(cfg["detector_window_sec"]) * fs)), 16)
    step = max(int(round(float(cfg["detector_step_sec"]) * fs)), 1)
    max_prefix = int(round(float(cfg["max_prefix_sec"]) * fs))
    threshold = float(cfg["harmonic_ratio_threshold"])
    min_gap = int(round(float(cfg["min_event_gap_sec"]) * fs))
    max_events = int(cfg["max_events_per_recording"])
    scored: list[tuple[int, float, float]] = []

    proxy = np.real(z_hp)
    for end in range(window, len(proxy) - max_prefix + 1, step):
        x = proxy[end - window:end]
        hr, f0 = autocorrelation_harmonic_ratio(x, fs)
        scored.append((end, hr, f0))

    candidates = [(end, hr, f0, False) for end, hr, f0 in scored if hr >= threshold]
    if not candidates and scored:
        top_k = max(int(cfg.get("fallback_top_k", 0)), 0)
        candidates = [(e, h, f, True) for e, h, f in sorted(scored, key=lambda t: t[1], reverse=True)[:top_k]]

    selected: list[tuple[int, float, float, bool]] = []
    for cand in sorted(candidates, key=lambda t: (-t[1], t[0])):
        end, hr, f0, fallback = cand
        if all(abs(end - prev[0]) >= min_gap for prev in selected):
            selected.append(cand)
        if len(selected) >= max_events:
            break
    selected.sort(key=lambda t: t[0])
    return [
        EventCandidate(start_sample=end, detector_end_sample=end, harmonic_ratio=float(hr), detector_f0_hz=float(f0), fallback=bool(fallback))
        for end, hr, f0, fallback in selected
    ]


def events_to_frame(recording_row: pd.Series, events: list[EventCandidate]) -> pd.DataFrame:
    rows = []
    for idx, event in enumerate(events):
        event_id = f"{recording_row.recording_id}__e{idx:03d}"
        rows.append(
            {
                "event_id": event_id,
                "recording_id": recording_row.recording_id,
                "subject_id": recording_row.subject_id,
                "species": recording_row.species,
                "path": recording_row.path,
                "start_sample": event.start_sample,
                "detector_end_sample": event.detector_end_sample,
                "detector_harmonic_ratio": event.harmonic_ratio,
                "detector_f0_hz": event.detector_f0_hz,
                "fallback_event": event.fallback,
            }
        )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)
