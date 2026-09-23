"""Day 28: stateless sepsis scoring API.

Design: the client sends the trailing window of hourly vitals; the server
holds NO session state (no per-patient memory, no sticky sessions). State
lives client-side (or in a lightweight cache in front of this service),
so any replica can score any request — horizontally scalable by default.

POST /score {"rows": [{"ICULOS": 12, "HR": 97.0, ...}, ...]}
  -> {"risk": 0.31, "alert": true, "hours_observed": 12, "threshold": 0.25}
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features import SIGNALS, add_window_features

MODEL_PATH = PROJECT_ROOT / "models" / "lgbm_windows.joblib"
THRESHOLD = 0.25  # Day-25 recall-0.8 operating point (0.2527)

app = FastAPI(title="Sepsis Early-Warning API")

model = joblib.load(MODEL_PATH)
# Column order straight from the fitted estimator — always matches training.
FEATURES = list(model.feature_names_in_)


class VitalRow(BaseModel):
    """One hourly measurement; any signal may be absent (None -> NaN)."""

    ICULOS: int
    HR: float | None = None
    O2Sat: float | None = None
    Temp: float | None = None
    SBP: float | None = None
    MAP: float | None = None
    DBP: float | None = None
    Resp: float | None = None
    Lactate: float | None = None
    WBC: float | None = None
    Creatinine: float | None = None
    Bilirubin_total: float | None = None
    Platelets: float | None = None
    Glucose: float | None = None


class ScoreRequest(BaseModel):
    """Trailing window + static patient context (the model trained on both).

    Static context is per-request data, not server state — the endpoint stays
    stateless. Units may be unknown (None -> NaN, which the trees handle).
    """

    rows: list[VitalRow] = Field(min_length=1)
    age: float
    gender: int
    unit1: float | None = None
    unit2: float | None = None
    hosp_adm_time: float | None = None


class ScoreResponse(BaseModel):
    risk: float
    alert: bool
    hours_observed: int
    threshold: float


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    try:
        df = pd.DataFrame([r.model_dump() for r in req.rows])
        df = df.sort_values("ICULOS").reset_index(drop=True)
        df["pid"] = "api"  # single-window scope; no cross-request state
        df["Age"] = req.age
        df["Gender"] = req.gender
        df["Unit1"] = req.unit1
        df["Unit2"] = req.unit2
        df["HospAdmTime"] = req.hosp_adm_time
        # All-None columns arrive as object dtype; the model needs floats.
        for col in SIGNALS + ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        feat = add_window_features(df)
        last = feat[FEATURES].iloc[[-1]]
        risk = float(model.predict_proba(last)[0, 1])
        return ScoreResponse(
            risk=risk,
            alert=risk >= THRESHOLD,
            hours_observed=len(df),
            threshold=THRESHOLD,
        )
    except Exception as e:  # noqa: BLE001 - any scoring failure -> 400, never 500
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
