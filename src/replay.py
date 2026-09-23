"""Day 27: streaming replay simulator.

Replays a patient's raw hourly vitals in order. At each hour *t* the trailing
window is rebuilt from rows with ICULOS <= *t* ONLY, then scored — exactly the
deployment constraint. Batch-scoring the full record can't prove this, because
nothing stops future rows from influencing past scores.

Output: models/replay_demo.json (risk trajectories + lead-time stats)
"""
import glob
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aggregate_eda import scan_file
from baseline import feature_columns
from features import add_window_features

DATA_DIR = PROJECT_ROOT / "data" / "training_setA"
MODEL_PATH = PROJECT_ROOT / "models" / "lgbm_windows.joblib"
OUT_PATH = PROJECT_ROOT / "models" / "replay_demo.json"

LABEL = "SepsisLabel"
THRESHOLD = 0.25  # Day-25 recall-0.8 operating point (0.2527)


def load_patient_raw(path):
    """Raw hourly frame for one patient (no features, no future)."""
    df = pd.read_csv(path, sep="|")
    return df.sort_values("ICULOS").reset_index(drop=True)


def replay_patient(df_raw, model, cols, threshold=THRESHOLD):
    """Score hour-by-hour; each step sees only the past. Returns trajectory."""
    traj = []
    for t in range(len(df_raw)):
        prefix = df_raw.iloc[: t + 1].copy()
        prefix["pid"] = "replay"
        feat = add_window_features(prefix)
        risk = float(model.predict_proba(feat[cols].iloc[[-1]])[0, 1])
        traj.append({"hour": int(df_raw.loc[t, "ICULOS"]), "risk": round(risk, 4)})
    return traj


def summarize(pid, df_raw, traj, threshold=THRESHOLD):
    """First alert, lead time vs clinical onset (labels are pre-shifted +6h)."""
    hours = df_raw.loc[df_raw[LABEL] == 1, "ICULOS"]
    onset_label = int(hours.min()) if len(hours) else None
    clinical_onset = onset_label + 6 if onset_label is not None else None
    alerts = [p for p in traj if p["risk"] >= threshold]
    first_alert = alerts[0]["hour"] if alerts else None
    lead = clinical_onset - first_alert if clinical_onset and first_alert else None
    return {
        "pid": pid,
        "n_hours": len(df_raw),
        "septic": onset_label is not None,
        "onset_label_hour": onset_label,
        "clinical_onset_hour": clinical_onset,
        "first_alert_hour": first_alert,
        "lead_time_hours": lead,
        "max_risk": max(p["risk"] for p in traj),
        "trajectory": traj,
    }


def pick_demo_patients():
    """One septic (onset 24-72h, room to warn) + one long clean stay."""
    septic_pid = clean_pid = None
    for fp in sorted(glob.glob(str(DATA_DIR / "*.psv"))):
        n_h, onset = scan_file(fp)
        pid = Path(fp).stem
        if septic_pid is None and onset is not None and 24 <= onset <= 72:
            septic_pid = pid
        if clean_pid is None and onset is None and n_h >= 72:
            clean_pid = pid
        if septic_pid and clean_pid:
            break
    return septic_pid, clean_pid


def main():
    model = joblib.load(MODEL_PATH)
    # Column spec from a real featurized row (same 96 cols the model trained on).
    probe = load_patient_raw(min(glob.glob(str(DATA_DIR / "*.psv"))))
    probe["pid"] = "probe"
    cols = feature_columns(add_window_features(probe))

    septic_pid, clean_pid = pick_demo_patients()
    print(f"demo patients: septic={septic_pid}, clean={clean_pid}")
    results = []
    for pid in (septic_pid, clean_pid):
        df_raw = load_patient_raw(DATA_DIR / f"{pid}.psv")
        traj = replay_patient(df_raw, model, cols)
        summary = summarize(pid, df_raw, traj)
        results.append(summary)
        print(
            f"{pid}: septic={summary['septic']} onset@{summary['clinical_onset_hour']} "
            f"first_alert@{summary['first_alert_hour']} lead={summary['lead_time_hours']}h "
            f"max_risk={summary['max_risk']}"
        )
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
