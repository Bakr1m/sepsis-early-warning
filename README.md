# Project 2: Sepsis Early-Warning System

**Days 21–30 | Healthcare ML Portfolio**

## Business Context

Sepsis mortality rises with every hour of delayed treatment. This project
replays ICU vitals hour-by-hour and scores deterioration risk continuously —
predicting onset **6 hours ahead** — so teams can start fluids, antibiotics,
and escalation *before* the crash instead of reacting to it.

## Dataset

- **Source**: PhysioNet/Computing in Cardiology Challenge 2019 (open access)
- **Scale**: training set A — 20,336 patients, 790,215 hourly rows, 41 columns
  per row (vitals, labs, demographics, `ICULOS`, `SepsisLabel`)
- **Target**: `SepsisLabel` at hour *t* (labels pre-shifted 6h: `1` means
  clinical onset at *t+6*); 8.8% of patients, **2.17% of rows** positive
- **Key challenges**: extreme imbalance, >90% missing labs, onset spread
  across the whole stay (median 30h), 21% onset within 6h of admission
- **Download** (gitignored): `wget -r -N -c -np
  https://physionet.org/files/challenge-2019/1.0.0/training/ -P data/`

## Approach

1. **EDA + window definition** (Day 21): file layout, prevalence, onset timing;
   predict onset 6h ahead from past-only data.
2. **Causal feature engineering** (Day 22): 13 signals × trailing 6h/24h
   mean+std, missingness fractions, `shift(1)` rate-of-change deltas = 91
   features; NaNs preserved, never filled.
3. **Baseline** (Day 23): Logistic Regression on windows, patient-ID split
   (row shuffling would leak patients' futures into training).
4. **Trees vs sequences** (Day 24): NaN-native LightGBM vs 1-layer LSTM on
   raw 24h sequences (all positives + 10x negatives, CPU-feasible).
5. **Rigorous evaluation** (Day 25): 5-fold GroupKFold by patient, OOF-based
   threshold analysis, gain feature importance.
6. **Tracking + robustness** (Day 26): MLflow runs; partial-record tests
   (sparse real-time hours must score, not crash).
7. **Replay simulator** (Day 27): hour-by-hour past-only scoring with
   future-truncation invariance tests.
8. **Serving + container** (Day 28): stateless FastAPI `POST /score`, 685 MB
   Docker image with bit-identical predictions.
9. **Dashboard + deploy** (Day 29): Streamlit risk-trend viewer.

## Results

| Model | ROC-AUC | PR-AUC | Notes |
|-------|---------|--------|-------|
| LR baseline (full test) | 0.7237 | 0.0060 | median-impute compromise |
| LightGBM (full test) | **0.7329** | **0.0078** | NaN-native, F1 doubled vs baseline |
| LightGBM (shared sample) | 0.7361 | **0.2439** | like-for-like vs LSTM |
| LSTM (shared sample) | 0.7389 | 0.2139 | ranking tie, loses on PR |

- **CV stability**: 5-fold GroupKFold mean ROC-AUC 0.7498 ± 0.012 (no fold collapses).
- **Operating point** (recall ≥ 0.8): threshold **0.25** → precision 0.004,
  **~1,072 alerts per 100 patient-days** — the alert-fatigue problem, quantified.
- **Top drivers**: Resp/Temp trailing means, Creatinine_mean_24h, HR/SBP means,
  `Platelets_miss_24h` (missingness vindicated); HospAdmTime/Unit1 flag a
  site-effect to validate cross-hospital.

## Limitations

1. **Single-source training**: set A only (two hospital systems); set B held out —
   cross-site validation still open, especially given admin-feature importance.
2. **No wall-clock timestamps**: calendar forward-chaining impossible; CV is
   patient-grouped with time-ordered rows (strongest available, documented gap).
3. **Retrospective labels**: Sepsis-3 from chart data; prospective bedside
   validation required before any clinical use.
4. **Alert load**: ~11 alerts/patient-day at recall 0.8 — undeployable without
   sustained-elevation rules (see Day-30 discussion).
5. **Sparse early windows over-alert**: fresh admissions with no labs score high;
   needs calibration work for hours 1–6.

## Ethical Considerations

- **Alert fatigue kills**: a system crying wolf 11×/day gets muted, then misses
  the real case — threshold + persistence rules are safety features, not tuning.
- **Human-in-the-loop**: decision support only; no autonomous treatment triggers.
- **Fairness**: audit across units, age, sex before deployment; site features in
  the model demand disparate-impact checks.
- **Validation gap**: research prototype on retrospective data, not a device.

## Project Structure

```
project2_sepsis/
├ data/                  # training_setA/*.psv, windows_setA.parquet (gitignored)
├ notebooks/             # 01_explore ... 10_wrapup (all execute clean)
├ src/
│   ├── aggregate_eda.py  # set-level stats -> models/eda_summary.json
│   ├── features.py       # causal rolling windows (91 feats)
│   ├── baseline.py       # LR baseline, patient-ID split
│   ├── lgbm.py           # NaN-native LightGBM + MLflow
│   ├── sequence.py       # LSTM on sampled 24h sequences + like-for-like compare
│   ├── evaluate.py       # GroupKFold CV, thresholds, importance + MLflow
│   ├── replay.py         # hour-by-hour past-only replay simulator
│   └── serve.py          # FastAPI app: POST /score (stateless)
├ api/main.py            # thin entrypoint
├ dashboard/app.py       # Streamlit risk-trend viewer
├ tests/                 # 30 hermetic tests (synthetic data only)
├ models/                # artifacts: *.joblib/*.pt/*.json (gitignored, via make train)
├ mlruns/                # MLflow tracking (gitignored)
├ example_window.json    # real 12h scoring payload (contract test fixture)
├ Dockerfile (685 MB) + requirements-serve.txt (serving deps only)
├ requirements.txt (full local env) / requirements-train.txt (torch+mlflow)
├ Makefile               # install / test / lint / train / serve / dashboard / docker-*
└ README.md
```

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # full local stack
# or: pip install -r requirements-serve.txt   # serving only (what Docker uses)

make test      # 30 hermetic tests (no data/ needed)
make train     # features -> baseline -> LightGBM (needs data/, ~10 min)
make serve     # API on :8000
make dashboard # Streamlit on :8501
```

## Run with Docker

```bash
docker pull bakr1m/sepsis-api:v1
docker run -p 8000:8000 bakr1m/sepsis-api:v1
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" -d @example_window.json
# -> {"risk":0.5437,"alert":true,"hours_observed":12,"threshold":0.25}
```

## Key Learnings

1. **Imbalance dominates everything** — 2.17% rows positive makes accuracy
   meaningless and PR-AUC the only honest metric.
2. **Causality is a code property** — trailing windows + truncation-invariance
   tests, not good intentions.
3. **Trees beat sequences on scarce positives + strong features** (PR 0.244 vs
   0.214); sequences need data scale to earn their complexity.
4. **Missingness is signal** (MNAR) — preserve + flag, never silently fill.
5. **Replay evaluates what batch metrics can't** — lead-time honesty,
   sparse-window behavior, alert dynamics.
6. **Stateless serving scales** — window travels with the request.
