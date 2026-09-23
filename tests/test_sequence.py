"""Hermetic tests for the LSTM pipeline + LightGBM smoke test (no data/)."""
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from src.features import SIGNALS
from src.sequence import LOOKBACK, SepsisLSTM, build_sequences, sample_rows


def make_patient(pid, n, label_at=None):
    data = {"pid": [pid] * n, "ICULOS": list(range(1, n + 1))}
    for col in SIGNALS:
        data[col] = [1.0] * n
    data["HR"] = [float(60 + i) for i in range(n)]  # ramp: trend signal
    data["SepsisLabel"] = [0] * n
    if label_at is not None:
        data["SepsisLabel"][label_at] = 1
    return pd.DataFrame(data)


def _stats():
    return {
        "mean": np.zeros(len(SIGNALS), dtype=np.float32),
        "std": np.ones(len(SIGNALS), dtype=np.float32),
    }


def test_sequence_shapes_and_masks():
    df = make_patient("p1", 5)
    by_pid = {"p1": df}
    X = build_sequences(df, by_pid, _stats())
    assert X.shape == (5, LOOKBACK, 2 * len(SIGNALS))
    n_sig = len(SIGNALS)
    # Last row: fully observed HR -> mask 1; padded prefix -> mask 0.
    assert X[4, -5:, n_sig:].min() == 1.0
    assert X[4, :-5, n_sig:].max() == 0.0


def test_sequence_causal():
    """Changing a future row must not alter earlier rows' sequences."""
    df = make_patient("p1", 6)
    by_pid = {"p1": df}
    before = build_sequences(df.iloc[:3], by_pid, _stats())
    df2 = df.copy()
    df2.loc[5, "HR"] = 999.0
    after = build_sequences(df2.iloc[:3], {"p1": df2}, _stats())
    np.testing.assert_array_equal(before, after)


def test_sample_rows_keeps_all_positives():
    df = pd.concat(
        [make_patient("p1", 30, label_at=29), make_patient("p2", 30)], ignore_index=True
    )
    samp = sample_rows(df, seed=0)
    assert (samp["SepsisLabel"] == 1).sum() == 1  # the single positive kept
    assert len(samp) == 1 + 10  # + 10x negatives


def test_lstm_forward_shape():
    net = SepsisLSTM()
    out = net(torch.zeros(4, LOOKBACK, 2 * len(SIGNALS)))
    assert out.shape == (4,)


def test_lgbm_fits_with_nan():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(60, 6), columns=[f"f{i}" for i in range(6)])
    X.iloc[::3, 0] = np.nan
    y = (X["f1"] > 0.5).astype(int)
    clf = lgb.LGBMClassifier(n_estimators=10, min_child_samples=5, verbose=-1)
    clf.fit(X, y)
    assert clf.predict_proba(X).shape == (60, 2)
