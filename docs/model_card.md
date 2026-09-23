# Model Card — Sepsis Early-Warning System v1.0.0

## Model details
- **Architecture:** LightGBM on 91 causal trailing-window features
  (13 signals × 6h/24h mean+std+miss + `shift(1)` deltas) + 5 static fields.
- **Artifact:** `models/lgbm_windows.joblib`; served statelessly via
  FastAPI `POST /score`; Docker `bakr1m/sepsis-api:v1` (685 MB).
- **Operating point:** threshold 0.25 (recall ≥ 0.8 on OOF predictions).

## Intended use
- **Decision support only:** hourly deterioration risk to prioritize review,
  never an autonomous trigger. **Out of scope:** diagnosis, treatment
  selection, non-ICU populations, use without clinician review.

## Training data
- PhysioNet Challenge 2019, training set A (20,336 ICU patients, 790,215
  hourly rows). 8.8% patients / 2.17% rows positive. Labels pre-shifted +6h.

## Evaluation
- 5-fold GroupKFold by patient: mean ROC-AUC 0.7498 ± 0.012, PR-AUC ~0.009.
- At recall 0.8: precision 0.004 → ~1,072 alerts / 100 patient-days.
- LSTM comparison (shared sample): ROC tie (0.739 vs 0.736), LGBM wins PR
  (0.244 vs 0.214). Retrospective only — no prospective validation.

## Limitations & ethics
- Trained on two hospital systems; admin-feature importance (Unit1,
  HospAdmTime) flags site effects — validate cross-site (set B) first.
- No wall-clock timestamps: no calendar forward-chaining possible.
- Sparse early hours over-alert; sustained-elevation rules required before
  any bedside use (alert fatigue is a safety issue).
- See README (Limitations, Ethical Considerations) for the full list.
