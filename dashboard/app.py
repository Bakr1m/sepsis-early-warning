"""Day 29: risk-trend dashboard over replayed patient trajectories.

Run:  .venv/bin/streamlit run dashboard/app.py
Shows the Day-27 replay trajectories (risk vs hour, alert threshold,
first-alert/lead-time markers) plus the underlying vitals — because a rising
trend signals deterioration before any threshold crossing does.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEMO_PATH = ROOT / "models" / "replay_demo.json"
DATA_DIR = ROOT / "data" / "training_setA"
THRESHOLD = 0.25

VITAL_PLOT_COLS = ["HR", "MAP", "Resp", "O2Sat"]


def load_demo():
    return json.loads(DEMO_PATH.read_text())


def trajectory_frame(patient):
    """Risk trajectory as an hour-indexed frame with threshold + alert flags."""
    df = pd.DataFrame(patient["trajectory"]).set_index("hour").sort_index()
    df["threshold"] = THRESHOLD
    df["alert"] = df["risk"] >= THRESHOLD
    return df


def alert_hours(frame):
    """Hours at or above threshold (sustained runs matter — see Day 27)."""
    return frame.index[frame["alert"]].tolist()


def vitals_frame(pid, n_hours=None, data_dir=DATA_DIR):
    """Raw vitals for context under the risk curve (NaNs = unmeasured hours)."""
    df = pd.read_csv(Path(data_dir) / f"{pid}.psv", sep="|")
    cols = ["ICULOS"] + [c for c in VITAL_PLOT_COLS if c in df.columns]
    df = df[cols].set_index("ICULOS").sort_index()
    return df if n_hours is None else df.head(n_hours)


def main():
    import streamlit as st

    st.title("Sepsis Early-Warning — Risk Trend")
    demo = load_demo()
    labels = {
        p["pid"]: f"{p['pid']} ({'septic' if p['septic'] else 'clean'})" for p in demo
    }
    pid = st.selectbox("Patient", list(labels), format_func=labels.get)
    patient = next(p for p in demo if p["pid"] == pid)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max risk", f"{patient['max_risk']:.2f}")
    c2.metric("First alert (h)", str(patient["first_alert_hour"]))
    c3.metric("Lead time (h)", str(patient["lead_time_hours"]))
    c4.metric("Clinical onset (h)", str(patient["clinical_onset_hour"]))

    traj = trajectory_frame(patient)
    st.subheader("Risk trajectory vs alert threshold")
    st.line_chart(traj[["risk", "threshold"]])
    st.caption(
        f"Red line: operating point {THRESHOLD} (Day-25 recall-0.8). "
        f"Alert hours: {alert_hours(traj)[:8]}"
        f"{' …' if len(alert_hours(traj)) > 8 else ''}"
    )

    st.subheader("Vitals underneath the curve")
    st.line_chart(vitals_frame(pid))
    st.caption("Gaps = unmeasured hours (NaN), not zeros.")


if __name__ == "__main__":
    main()
