"""API contract tests (in-process TestClient; example_window.json is real data)."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.serve import THRESHOLD, app

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
