import pandas as pd
from qtrc_stop.stopping import Policy, apply_policy
from qtrc_stop.taxonomy import SPECIES


def test_policy_stops():
    rows = []
    for d, p in [(0.1, 0.6), (0.2, 0.95), (2.0, 0.99)]:
        row = {"event_id":"e1","subject_id":"s1","species":"apis_mellifera","duration_sec":d,"harmonic_ratio":0.8,"harmonic_energy_ratio":0.5,"f0_delta_hz":1.0}
        for s in SPECIES: row[f"p_{s}"] = (p if s == "apis_mellifera" else (1-p)/4)
        rows.append(row)
    policy = Policy(0.9,1.0,0.0,0.0,1e9,1,0.1,"empirical",0,0,0)
    out = apply_policy(pd.DataFrame(rows), [0.1,0.2,2.0], policy)
    assert out.iloc[0].stop_duration_sec == 0.2
