"""Day 24a: LightGBM on engineered windows (same split as the LR baseline).

Key difference from the baseline: NO imputation — LightGBM splits NaN
natively, so the missingness signal from Day 22 survives intact.

Output: models/lgbm_windows.joblib + models/lgbm_metrics.json
"""
import json
import os
import sys
from pathlib import Path

# MLflow 3.x gates the local file-store backend: opt in to keep mlruns/.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from baseline import exclude_post_onset, feature_columns, split_by_patient

WINDOWS_PATH = PROJECT_ROOT / "data" / "windows_setA.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "lgbm_windows.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "lgbm_metrics.json"
# Plain directory path (not a file: URI) — avoids URI-parsing issues in MLflow 3.x.
TRACKING_URI = str(PROJECT_ROOT / "mlruns")
EXPERIMENT = "sepsis_early_warning"

LABEL = "SepsisLabel"

PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}


def metrics_at(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    df = pd.read_parquet(WINDOWS_PATH)
    df = exclude_post_onset(df)
    train, test = split_by_patient(df)
    cols = feature_columns(train)
    X_train, y_train = train[cols], train[LABEL].astype(int)
    X_test, y_test = test[cols], test[LABEL].astype(int)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="lgbm_windows"):
        mlflow.log_params(PARAMS)
        mlflow.log_param("no_imputation", True)
        mlflow.log_param("n_train_rows", len(train))

        clf = lgb.LGBMClassifier(**PARAMS)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        m = metrics_at(y_test, proba)
        mlflow.log_metric("roc_auc", m["roc_auc"])
        mlflow.log_metric("pr_auc", m["pr_auc"])
        mlflow.log_metric("f1", m["f1"])
        mlflow.lightgbm.log_model(clf, "model")

    out = {
        "n_train_rows": len(train),
        "n_test_rows": len(test),
        "n_features": len(cols),
        "no_imputation": True,
        **{f"{k}_at_0.5": v for k, v in m.items() if k != "confusion"},
        "confusion_at_0.5": m["confusion"],
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
