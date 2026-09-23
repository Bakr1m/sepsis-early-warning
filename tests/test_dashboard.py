"""Hermetic dashboard helper tests (synthetic trajectory + tmp .psv)."""
import pandas as pd

from dashboard.app import alert_hours, trajectory_frame, vitals_frame

PATIENT = {
    "pid": "pX",
    "trajectory": [
        {"hour": 1, "risk": 0.10},
        {"hour": 2, "risk": 0.30},
        {"hour": 3, "risk": 0.20},
    ],
}


def test_trajectory_frame_marks_alerts():
    frame = trajectory_frame(PATIENT)
    assert list(frame.index) == [1, 2, 3]
    assert frame["threshold"].eq(0.25).all()
    assert frame["alert"].tolist() == [False, True, False]


def test_alert_hours_lists_crossings():
    assert alert_hours(trajectory_frame(PATIENT)) == [2]


def test_vitals_frame_reads_psv(tmp_path):
    df = pd.DataFrame(
        {
            "ICULOS": [1, 2],
            "HR": [80.0, 90.0],
            "MAP": [70.0, None],
            "Resp": [16.0, 18.0],
            "O2Sat": [98.0, 97.0],
        }
    )
    df.to_csv(tmp_path / "pX.psv", sep="|", index=False)
    out = vitals_frame("pX", data_dir=tmp_path)
    assert list(out.index) == [1, 2]
    assert out.loc[1, "HR"] == 80.0
    assert pd.isna(out.loc[2, "MAP"])  # gaps preserved, not zero-filled
