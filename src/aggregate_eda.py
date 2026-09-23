"""Day 21 EDA aggregation: patient/row-level sepsis stats over training set A.

Reads only ICULOS + SepsisLabel per file (fast pure-python parse).
Writes models/eda_summary.json and prints the summary.
"""
import csv
import glob
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "training_setA"
OUT_PATH = PROJECT_ROOT / "models" / "eda_summary.json"


def scan_file(path):
    """Return (n_hours, onset_hour or None) for one patient .psv."""
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter="|")
        header = next(reader)
        icu_idx = header.index("ICULOS")
        lab_idx = header.index("SepsisLabel")
        n_hours = 0
        onset_hour = None
        for row in reader:
            if not row:
                continue
            n_hours += 1
            if onset_hour is None and row[lab_idx] == "1":
                onset_hour = int(float(row[icu_idx]))
    return n_hours, onset_hour


def main():
    files = sorted(glob.glob(str(DATA_DIR / "*.psv")))
    if not files:
        sys.exit(f"no .psv files in {DATA_DIR} — mirror set A first (see README)")
    t0 = time.time()
    n_patients = len(files)
    n_septic = 0
    n_rows = 0
    onset_hours = []
    for i, fp in enumerate(files, 1):
        n_h, onset = scan_file(fp)
        n_rows += n_h
        if onset is not None:
            n_septic += 1
            onset_hours.append(onset)
        if i % 2000 == 0:
            print(f"  {i}/{n_patients} files…", flush=True)
    # NOTE: positive-row count needs label sums; recompute cheaply below.
    pos_rows = 0
    for fp in files:
        with open(fp) as f:
            rdr = csv.reader(f, delimiter="|")
            header = next(rdr)
            li = header.index("SepsisLabel")
            pos_rows += sum(1 for row in rdr if row and row[li] == "1")

    summary = {
        "n_patients": n_patients,
        "n_septic_patients": n_septic,
        "patient_prevalence": round(n_septic / n_patients, 4),
        "n_hourly_rows": n_rows,
        "n_positive_rows": pos_rows,
        "row_positive_rate": round(pos_rows / n_rows, 5),
        "onset_hour_min": min(onset_hours),
        "onset_hour_median": sorted(onset_hours)[len(onset_hours) // 2],
        "onset_hour_max": max(onset_hours),
        "onset_within_6h_pct": round(
            sum(1 for h in onset_hours if h <= 6) / len(onset_hours) * 100, 1
        ),
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
