"""Day 23: baseline Logistic Regression on engineered windows.

- Drops post-onset hours (model must fire *before* onset, never after).
- Splits by patient ID, stratified by patient-level sepsis flag
  (row shuffling would leak a patient's future into training).
- Median-impute + scale + class-balanced LR (linear models need dense input;
  trees on Day 24 will use the NaNs natively).

Output: models/lr_baseline.joblib + models/baseline_metrics.json
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WINDOWS_PATH = PROJECT_ROOT / "data" / "windows_setA.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "lr_baseline.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "baseline_metrics.json"

LABEL = "SepsisLabel"
EXCLUDE = {"pid", "ICULOS", LABEL}
TEST_SIZE = 0.2
SEED = 42


def exclude_post_onset(df):
    """Keep rows up to and including onset; drop strictly post-onset hours."""
    onset = df.loc[df[LABEL] == 1].groupby("pid")["ICULOS"].min()
    onset_per_row = df["pid"].map(onset)
    return df[(onset_per_row.isna()) | (df["ICULOS"] <= onset_per_row)].copy()


def split_by_patient(df, test_size=TEST_SIZE, seed=SEED):
    """Patient-ID split, stratified by patient-level sepsis flag."""
    flag = df.groupby("pid")[LABEL].max()
    septic = flag[flag == 1].index.tolist()
    nonseptic = flag[flag == 0].index.tolist()
    test_pids = []
    for group in (septic, nonseptic):
        _, test = train_test_split(group, test_size=test_size, random_state=seed)
        test_pids.extend(test)
    test_mask = df["pid"].isin(test_pids)
    return df[~test_mask].copy(), df[test_mask].copy()


def feature_columns(df):
    """Engineered window features + static demographics (no IDs/labels)."""
    static = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]
    engineered = [
        c
        for c in df.columns
        if "_mean_" in c or "_std_" in c or "_miss_" in c or "_delta_" in c
    ]
    return engineered + [c for c in static if c in df.columns]


def main():
    df = pd.read_parquet(WINDOWS_PATH)
    df = exclude_post_onset(df)
    train, test = split_by_patient(df)
    cols = feature_columns(train)
    X_train, y_train = train[cols], train[LABEL].astype(int)
    X_test, y_test = test[cols], test[LABEL].astype(int)

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=SEED
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    metrics = {
        "n_train_rows": len(train),
        "n_test_rows": len(test),
        "n_train_patients": int(train["pid"].nunique()),
        "n_test_patients": int(test["pid"].nunique()),
        "test_positive_rate": round(float(y_test.mean()), 5),
        "n_features": len(cols),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "f1_at_0.5": round(float(f1_score(y_test, pred)), 4),
        "precision_at_0.5": round(float(precision_score(y_test, pred)), 4),
        "recall_at_0.5": round(float(recall_score(y_test, pred)), 4),
        "confusion_at_0.5": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
