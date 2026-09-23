"""Hermetic tests for causal window features (synthetic patients, no data/).

Core invariant under test: every feature at hour *t* uses only rows <= *t*.
"""
import numpy as np
import pandas as pd

from src.features import SIGNALS, add_window_features


def make_patient(pid, values, start=1):
    """Build a minimal patient frame; unspecified signals default to NaN."""
    n = max(len(v) for v in values.values())
    data = {"pid": [pid] * n, "ICULOS": list(range(start, start + n))}
    for col in SIGNALS:
        data[col] = list(values.get(col, [np.nan] * n))
    return pd.DataFrame(data)


def test_causal_no_future_leak():
    """A spike at t=5 must not affect features at t<=4."""
    df = make_patient("p1", {"HR": [10, 10, 10, 10, 100, 100]})
    out = add_window_features(df)
    assert out.loc[3, "HR_mean_6h"] == 10.0
    assert out.loc[4, "HR_mean_6h"] == (10 * 4 + 100) / 5
    assert out.loc[4, "HR_delta_6h"] == 100 - 10.0


def test_first_row_invariants():
    """Row 0: mean == value, std == 0, no missingness, delta undefined."""
    out = add_window_features(make_patient("p1", {"HR": [72.0]}))
    assert out.loc[0, "HR_mean_6h"] == 72.0
    assert out.loc[0, "HR_std_6h"] == 0.0
    assert out.loc[0, "HR_miss_6h"] == 0.0
    assert np.isnan(out.loc[0, "HR_delta_6h"])


def test_all_nan_signal():
    """Fully unmeasured signal: NaN stats, miss fraction 1, std 0."""
    out = add_window_features(make_patient("p1", {"Lactate": [np.nan] * 8}))
    assert np.isnan(out.loc[7, "Lactate_mean_6h"])
    assert out.loc[7, "Lactate_miss_6h"] == 1.0
    assert out.loc[7, "Lactate_std_6h"] == 0.0
    assert np.isnan(out.loc[7, "Lactate_delta_6h"])


def test_no_cross_patient_leak():
    """Rolling windows reset at patient boundaries."""
    a = make_patient("pA", {"HR": [100.0] * 6})
    b = make_patient("pB", {"HR": [60.0] * 3})
    out = add_window_features(pd.concat([a, b], ignore_index=True))
    b0 = out[out["pid"] == "pB"].iloc[0]
    assert b0["HR_mean_6h"] == 60.0
    assert b0["HR_mean_24h"] == 60.0


def test_missing_fraction_exact():
    """2 of 6 observed in the trailing window -> miss == 4/6."""
    vals = [1.0, np.nan, np.nan, 3.0, np.nan, np.nan]
    out = add_window_features(make_patient("p1", {"WBC": vals}))
    assert out.loc[5, "WBC_miss_6h"] == 4 / 6
    assert out.loc[5, "WBC_mean_6h"] == (1.0 + 3.0) / 2
