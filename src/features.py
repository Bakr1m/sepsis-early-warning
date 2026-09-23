"""Day 22: causal time-series feature engineering for sepsis windows.

Every feature at hour *t* uses only data with ICULOS <= *t* (trailing rolling
windows plus shift(1) baselines), so no future information can leak.
NaNs are preserved (trees handle them); missingness gets explicit features.

Output: data/windows_setA.parquet + models/features_summary.json
"""
import glob
import json
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "training_setA"
OUT_PATH = PROJECT_ROOT / "data" / "windows_setA.parquet"
SUMMARY_PATH = PROJECT_ROOT / "models" / "features_summary.json"

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
LABS = ["Lactate", "WBC", "Creatinine", "Bilirubin_total", "Platelets", "Glucose"]
SIGNALS = VITALS + LABS
WINDOWS = [6, 24]
DEMOGRAPHICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]
READ_COLS = ["ICULOS", "SepsisLabel"] + DEMOGRAPHICS + SIGNALS

CHUNK_PATIENTS = 5000


def _rolled(grouped, window, how):
    """Groupby-rolling aggregation realigned to the frame's row index."""
    rolled = getattr(grouped.rolling(window=window, min_periods=1), how)()
    return rolled.reset_index(level=0, drop=True)


def add_window_features(frame):
    """Add trailing-window features. `frame` needs pid + ICULOS + signals.

    Chunks must always contain whole patients — rolling never crosses a
    pid boundary because every operation is grouped by pid.
    """
    frame = frame.sort_values(["pid", "ICULOS"]).reset_index(drop=True)
    by_pid = frame["pid"]
    # Collect first, concat once: 91 sequential inserts fragment the frame
    # (and log PerformanceWarnings); a single concat avoids both.
    new_cols = {}
    for col in SIGNALS:
        s = frame[col]
        grouped = s.groupby(by_pid, sort=False)
        for w in WINDOWS:
            new_cols[f"{col}_mean_{w}h"] = _rolled(grouped, w, "mean")
            new_cols[f"{col}_std_{w}h"] = _rolled(grouped, w, "std").fillna(0.0)
            na = s.isna().astype("float32")
            new_cols[f"{col}_miss_{w}h"] = _rolled(
                na.groupby(by_pid, sort=False), w, "mean"
            )
        # Rate of change: current value vs trailing-6h baseline EXCLUDING
        # the current hour (shift(1) keeps it strictly backward-looking).
        baseline = _rolled(grouped.shift(1).groupby(by_pid, sort=False), 6, "mean")
        new_cols[f"{col}_delta_6h"] = s - baseline
    return pd.concat([frame, pd.DataFrame(new_cols, index=frame.index)], axis=1)


def read_chunk(paths):
    """Read patient files into one frame with a pid column."""
    frames = []
    for p in paths:
        df = pd.read_csv(p, sep="|", usecols=READ_COLS)
        df["pid"] = Path(p).stem
        frames.append(df)
    chunk = pd.concat(frames, ignore_index=True)
    for col in SIGNALS:
        chunk[col] = chunk[col].astype("float32")
    return chunk


def main():
    t0 = time.time()
    paths = sorted(glob.glob(str(DATA_DIR / "*.psv")))
    assert paths, f"no .psv files in {DATA_DIR}"
    chunks = []
    for i in range(0, len(paths), CHUNK_PATIENTS):
        c = add_window_features(read_chunk(paths[i : i + CHUNK_PATIENTS]))
        chunks.append(c)
        print(f"  patients {i + 1}-{min(i + CHUNK_PATIENTS, len(paths))}", flush=True)
    out = pd.concat(chunks, ignore_index=True)
    feat_cols = [
        c
        for c in out.columns
        if "_mean_" in c or "_std_" in c or "_miss_" in c or "_delta_" in c
    ]
    out[feat_cols] = out[feat_cols].astype("float32")
    out.to_parquet(OUT_PATH, index=False)
    summary = {
        "n_patients": len(paths),
        "n_rows": len(out),
        "n_features": len(feat_cols),
        "n_columns_total": len(out.columns),
        "raw_nan_rate_top": {
            col: round(float(out[col].isna().mean()), 4)
            for col in sorted(SIGNALS, key=lambda c: out[c].isna().mean(), reverse=True)[:5]
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
