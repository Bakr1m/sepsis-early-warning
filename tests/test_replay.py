"""Replay invariance: truncating future hours must not change past scores.

This is the deployment constraint as an executable assertion — the model
under replay may only ever see the past.
"""
import lightgbm as lgb
import pandas as pd

from src.features import SIGNALS, add_window_features
from src.replay import replay_patient


def make_patient(pid, n, hot_at=()):
    data = {"ICULOS": list(range(1, n + 1))}
    for col in SIGNALS:
        data[col] = [70.0] * n
    data["HR"] = [60.0 + i for i in range(n)]
    data["SepsisLabel"] = [1 if t in hot_at else 0 for t in range(1, n + 1)]
    return pd.DataFrame(data)


def feature_names(df):
    return [
        c
        for c in df.columns
        if "_mean_" in c or "_std_" in c or "_miss_" in c or "_delta_" in c
    ]


def tiny_model():
    frames = []
    for pid in ("a", "b"):
        df = make_patient(pid, 12, hot_at={10, 11, 12})
        df["pid"] = pid
        frames.append(df)
    train = add_window_features(pd.concat(frames, ignore_index=True))
    cols = feature_names(train)
    clf = lgb.LGBMClassifier(n_estimators=10, min_child_samples=2, verbose=-1)
    clf.fit(train[cols], train["SepsisLabel"])
    return clf, cols


def test_replay_truncation_invariance():
    """Scores for hours 1..T are identical with/without hours T+1.. present."""
    clf, cols = tiny_model()
    full = make_patient("c", 10)
    traj_full = replay_patient(full, clf, cols)
    traj_trunc = replay_patient(full.iloc[:6].copy(), clf, cols)
    assert len(traj_full) == 10 and len(traj_trunc) == 6
    assert [p["hour"] for p in traj_trunc] == [1, 2, 3, 4, 5, 6]
    assert [p["risk"] for p in traj_trunc] == [p["risk"] for p in traj_full[:6]]


def test_replay_trajectory_shape():
    clf, cols = tiny_model()
    traj = replay_patient(make_patient("c", 4), clf, cols)
    assert [p["hour"] for p in traj] == [1, 2, 3, 4]
    assert all(0.0 <= p["risk"] <= 1.0 for p in traj)
