"""Day 25: time-respecting evaluation of the winning LightGBM.

- GroupKFold by patient (stratified): no pid in two folds, rows stay
  time-ordered within each stay, so no future fold ever validates the past
  of the same patient. (The files carry no wall-clock admission timestamps,
  so calendar forward-chaining isn't possible — this boundary is documented
  in the notebook rather than faked.)
- Out-of-fold predictions -> deployment-threshold analysis (recall-anchored
  operating point + alert load in alerts per 100 patient-days).
- Gain-based feature importance for the winning model.

Output: models/cv_metrics.json + models/thresholds.json
        + models/feature_importance.json
"""
import json
import os
import sys
from pathlib import Path

# MLflow 3.x gates the local file-store backend: opt in to keep mlruns/.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from baseline import exclude_post_onset, feature_columns

WINDOWS_PATH = PROJECT_ROOT / "data" / "windows_setA.parquet"
CV_PATH = PROJECT_ROOT / "models" / "cv_metrics.json"
THR_PATH = PROJECT_ROOT / "models" / "thresholds.json"
IMP_PATH = PROJECT_ROOT / "models" / "feature_importance.json"
FINAL_PATH = PROJECT_ROOT / "models" / "lgbm_final.joblib"
# Plain directory path (not a file: URI) — avoids URI-parsing issues in MLflow 3.x.
TRACKING_URI = str(PROJECT_ROOT / "mlruns")
EXPERIMENT = "sepsis_early_warning"

LABEL = "SepsisLabel"
N_FOLDS = 5
SEED = 42
TREES = 150


def assign_folds(pids, flags, n_folds=N_FOLDS, seed=SEED):
    """Stratified patient-to-fold assignment (round-robin per class)."""
    rng = np.random.RandomState(seed)
    fold_of = {}
    for cls in (0, 1):
        group = [p for p, f in zip(pids, flags) if f == cls]
        rng.shuffle(group)
        for i, p in enumerate(group):
            fold_of[p] = i % n_folds
    return fold_of


def precision_recall_at(y_true, proba, threshold):
    pred = proba >= threshold
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": threshold,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "alert_rate": round(float(pred.mean()), 5),
    }


def threshold_for_recall(y_true, proba, target=0.8):
    """Highest threshold still achieving >= target recall on these scores."""
    order = np.argsort(-proba)
    y_sorted = np.asarray(y_true)[order]
    p_sorted = np.asarray(proba)[order]
    cum_tp = np.cumsum(y_sorted)
    recall = cum_tp / cum_tp[-1]
    ok = np.nonzero(recall >= target)[0]
    return float(p_sorted[ok[0]])


def train_fold(X_tr, y_tr):
    clf = lgb.LGBMClassifier(
        n_estimators=TREES,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
        verbose=-1,
    )
    clf.fit(X_tr, y_tr)
    return clf


def main():
    df = pd.read_parquet(WINDOWS_PATH)
    df = exclude_post_onset(df).reset_index(drop=True)  # positional OOF needs 0..N-1
    cols = feature_columns(df)
    pids = df["pid"].unique().tolist()
    flags = df.groupby("pid")[LABEL].max().reindex(pids).tolist()
    fold_of = assign_folds(pids, flags)
    df = df.copy()
    df["fold"] = df["pid"].map(fold_of)

    oof = np.zeros(len(df))
    fold_metrics = []
    for k in range(N_FOLDS):
        tr = df[df["fold"] != k]
        va = df[df["fold"] == k]
        clf = train_fold(tr[cols], tr[LABEL].astype(int))
        p = clf.predict_proba(va[cols])[:, 1]
        oof[va.index] = p
        fold_metrics.append(
            {
                "fold": k,
                "n_patients": int(va["pid"].nunique()),
                "roc_auc": round(float(roc_auc_score(va[LABEL], p)), 4),
                "pr_auc": round(float(average_precision_score(va[LABEL], p)), 4),
            }
        )
        print(f"  fold {k}: ROC-AUC={fold_metrics[-1]['roc_auc']} "
              f"PR-AUC={fold_metrics[-1]['pr_auc']}", flush=True)

    y = df[LABEL].astype(int).to_numpy()
    rocs = [f["roc_auc"] for f in fold_metrics]
    prs = [f["pr_auc"] for f in fold_metrics]
    cv = {
        "folds": fold_metrics,
        "mean_roc_auc": round(float(np.mean(rocs)), 4),
        "std_roc_auc": round(float(np.std(rocs)), 4),
        "mean_pr_auc": round(float(np.mean(prs)), 4),
        "std_pr_auc": round(float(np.std(prs)), 4),
        "oof_roc_auc": round(float(roc_auc_score(y, oof)), 4),
        "oof_pr_auc": round(float(average_precision_score(y, oof)), 4),
    }
    CV_PATH.write_text(json.dumps(cv, indent=2))

    op = threshold_for_recall(y, oof, target=0.8)
    grid = [precision_recall_at(y, oof, t) for t in (0.1, 0.2, 0.3, 0.5, 0.7, op)]
    for g in grid:  # rows are patient-hours -> alerts per 100 patient-days
        g["alerts_per_100_patient_days"] = round(g["alert_rate"] * 24 * 100, 1)
    THR_PATH.write_text(json.dumps({"operating_point": op, "grid": grid}, indent=2))

    final = train_fold(df[cols], y)
    imp = pd.Series(final.booster_.feature_importance(importance_type="gain"), index=cols)
    imp = imp.sort_values(ascending=False)
    IMP_PATH.write_text(
        json.dumps({k: round(float(v), 1) for k, v in imp.head(20).items()}, indent=2)
    )
    joblib.dump(final, FINAL_PATH)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="lgbm_groupkfold_cv"):
        mlflow.log_param("n_folds", N_FOLDS)
        mlflow.log_param("trees_per_fold", TREES)
        mlflow.log_metric("mean_roc_auc", cv["mean_roc_auc"])
        mlflow.log_metric("mean_pr_auc", cv["mean_pr_auc"])
        mlflow.log_metric("oof_roc_auc", cv["oof_roc_auc"])
        mlflow.log_metric("oof_pr_auc", cv["oof_pr_auc"])
        mlflow.log_metric("operating_point_recall_0.8", op)
    print(json.dumps(cv, indent=2))
    print(f"operating point (recall>=0.8): {op:.4f}")
    print("top-5:", list(imp.head(5).index))
    print(f"saved -> {CV_PATH}, {THR_PATH}, {IMP_PATH}")


if __name__ == "__main__":
    main()
