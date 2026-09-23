"""Hermetic tests for baseline data prep (synthetic patients, no data/)."""
import pandas as pd

from src.baseline import exclude_post_onset, split_by_patient


def make_frame():
    # pA septic with onset at t=3 (6 rows); pB clean (4 rows); pC septic at t=1.
    rows = []
    for pid, labels in {
        "pA": [0, 0, 1, 1, 1, 1],
        "pB": [0, 0, 0, 0],
        "pC": [1, 1],
    }.items():
        for t, lab in enumerate(labels, start=1):
            rows.append({"pid": pid, "ICULOS": t, "SepsisLabel": lab, "f": float(t)})
    return pd.DataFrame(rows)


def test_exclude_post_onset_keeps_onset_row():
    out = exclude_post_onset(make_frame())
    assert len(out[out["pid"] == "pA"]) == 3  # t<=3 kept, t>3 dropped
    assert len(out[out["pid"] == "pB"]) == 4  # clean patient untouched
    assert len(out[out["pid"] == "pC"]) == 1  # onset at t=1: only t=1 kept
    assert out["SepsisLabel"].sum() == 2  # one positive row per septic patient


def test_split_by_patient_no_overlap():
    # 10 clean + 4 septic patients so stratified splitting is well-defined.
    rows = []
    for i in range(10):
        for t in range(1, 5):
            rows.append({"pid": f"n{i}", "ICULOS": t, "SepsisLabel": 0, "f": 1.0})
    for i in range(4):
        for t, lab in enumerate([0, 0, 1, 1], start=1):
            rows.append({"pid": f"s{i}", "ICULOS": t, "SepsisLabel": lab, "f": 2.0})
    df = exclude_post_onset(pd.DataFrame(rows))
    train, test = split_by_patient(df, test_size=0.25, seed=42)
    assert set(train["pid"]).isdisjoint(set(test["pid"]))
    assert len(train) + len(test) == len(df)
    assert len(train) > 0 and len(test) > 0
    # Stratification: both splits contain septic and clean patients.
    assert train["SepsisLabel"].max() == 1 and test["SepsisLabel"].max() == 1
