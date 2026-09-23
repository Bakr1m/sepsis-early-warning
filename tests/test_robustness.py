"""Robustness: partial/incomplete vitals must never crash the pipeline.

Real-time streams routinely arrive with missing recent readings (a lab not
yet resulted, a sensor detached). These tests prove a sparse patient-hour
flows through feature building AND scoring with a defined output.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import SIGNALS, add_window_features


def make_sparse_patient(pid, n_hours):
    """Vitals only, all labs missing — the classic just-admitted patient."""
    vitals = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
    data = {"pid": [pid] * n_hours, "ICULOS": list(range(1, n_hours + 1))}
    for col in SIGNALS:
        if col in vitals:
            data[col] = [70.0 + i for i in range(n_hours)]
        else:
            data[col] = [np.nan] * n_hours
    data["SepsisLabel"] = [0] * n_hours
    return pd.DataFrame(data)


def feature_names(df):
    return [
        c
        for c in df.columns
        if "_mean_" in c or "_std_" in c or "_miss_" in c or "_delta_" in c
    ]


def test_partial_record_features_defined():
    """3-hour, labs-missing stay: vitals finite, labs flagged, shape kept."""
    out = add_window_features(make_sparse_patient("p1", 3))
    assert len(out) == 3
    assert np.isfinite(out.loc[2, "HR_mean_6h"])
    assert (out["Lactate_miss_6h"] == 1.0).all()
    assert out["Lactate_mean_6h"].isna().all()  # never invented


def test_single_hour_stay_no_crash():
    """A patient with one recorded hour still produces a full feature row."""
    out = add_window_features(make_sparse_patient("p1", 1))
    assert len(out) == 1
    assert out.loc[0, "HR_mean_6h"] == 70.0
    assert np.isfinite(out.loc[0, "HR_std_6h"])


def test_partial_record_end_to_end_scores():
    """Tiny model trained on synthetic windows scores a partial row finitely."""
    rng = np.random.RandomState(0)
    frames = []
    for i in range(30):
        df = make_sparse_patient(f"p{i}", 6)
        df["SepsisLabel"] = (rng.rand(6) < 0.2).astype(int).tolist()
        frames.append(df)
    train = add_window_features(pd.concat(frames, ignore_index=True))
    cols = feature_names(train)
    clf = lgb.LGBMClassifier(n_estimators=10, min_child_samples=2, verbose=-1)
    clf.fit(train[cols], train["SepsisLabel"])

    partial = add_window_features(make_sparse_patient("new", 2))
    proba = clf.predict_proba(partial[cols])[:, 1]
    assert len(proba) == 2
    assert np.isfinite(proba).all()
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_all_signals_missing_row_scores():
    """Even a fully unmeasured hour gets a defined (not crashing) output."""
    df = make_sparse_patient("p1", 4)
    for col in SIGNALS:
        df[col] = np.nan
    out = add_window_features(df)
    assert len(out) == 4
    assert (out["HR_miss_6h"] == 1.0).all()
