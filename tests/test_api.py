"""API contract tests (in-process TestClient; example_window.json is real data).

Hermetic by design: a tiny LGBM stand-in replaces the production weights
(no models/ needed — that dir is gitignored), so these run anywhere.
Production wiring (real weights file) is verified by the Docker parity check,
not unit tests.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import serve
from src.features import SIGNALS, add_window_features
from src.serve import THRESHOLD, app


@pytest.fixture(autouse=True)
def tiny_model_stand_in():
    """Train a 10-tree LGBM on synthetic windows; no disk artifacts touched."""
    rng = np.random.RandomState(0)
    frames = []
    for i in range(20):
        n = 6
        data = {"pid": [f"p{i}"] * n, "ICULOS": list(range(1, n + 1))}
        for col in SIGNALS:
            data[col] = rng.rand(n) * (100 if col == "HR" else 10)
        data["SepsisLabel"] = (rng.rand(n) < 0.3).astype(int).tolist()
        frames.append(pd.DataFrame(data))
    train = add_window_features(pd.concat(frames, ignore_index=True))
    cols = [c for c in train.columns if "_mean_" in c or "_miss_" in c]
    clf = lgb.LGBMClassifier(n_estimators=10, min_child_samples=2, verbose=-1)
    clf.fit(train[cols], train["SepsisLabel"])
    serve._model, serve._FEATURES = clf, cols
    yield
    serve._model, serve._FEATURES = None, None


client = TestClient(app)
EXAMPLE = json.loads(Path("example_window.json").read_text())


def test_health():
    assert client.get("/health").json() == {"status": "healthy"}


def test_score_real_window():
    """A real 12h trailing window scores a finite risk with observed=12."""
    r = client.post("/score", json=EXAMPLE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["risk"] <= 1.0
    assert body["hours_observed"] == 12
    assert body["alert"] == (body["risk"] >= THRESHOLD)
    assert body["threshold"] == THRESHOLD


def test_score_single_sparse_row():
    """One near-empty hour (deployment frontier) still scores, no crash."""
    payload = {"rows": [{"ICULOS": 1, "HR": 88.0}], "age": 60.0, "gender": 0}
    r = client.post("/score", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["hours_observed"] == 1


def test_score_empty_rejected():
    assert client.post("/score", json={"rows": []}).status_code == 422


def test_score_unsorted_rows_sorted():
    """Out-of-order hours are sorted server-side; risk matches ordered call."""
    ctx = {k: EXAMPLE[k] for k in ("age", "gender", "unit1", "unit2", "hosp_adm_time")}
    a = client.post("/score", json={**ctx, "rows": EXAMPLE["rows"]}).json()
    b = client.post("/score", json={**ctx, "rows": EXAMPLE["rows"][::-1]}).json()
    assert a["risk"] == b["risk"]
