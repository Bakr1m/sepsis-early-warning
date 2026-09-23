"""Day 24b: LSTM on raw 24h sequences, compared like-for-like with LightGBM.

CPU-feasible design: train on ALL positive hours + 10x random negatives
(same patient-ID split as the baseline). The saved LightGBM is scored on the
identical sampled test set, so the comparison is apples-to-apples.

Output: models/lstm_seq.pt + models/lstm_metrics.json + models/compare_metrics.json
"""
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from baseline import exclude_post_onset, feature_columns, split_by_patient
from features import SIGNALS
from lgbm import metrics_at

WINDOWS_PATH = PROJECT_ROOT / "data" / "windows_setA.parquet"
LGBM_PATH = PROJECT_ROOT / "models" / "lgbm_windows.joblib"
LSTM_PATH = PROJECT_ROOT / "models" / "lstm_seq.pt"
METRICS_PATH = PROJECT_ROOT / "models" / "lstm_metrics.json"
COMPARE_PATH = PROJECT_ROOT / "models" / "compare_metrics.json"

LABEL = "SepsisLabel"
LOOKBACK = 24
N_CHANNELS = 2 * len(SIGNALS)
NEG_MULTIPLE = 10
EPOCHS = 5
BATCH = 256
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(os.cpu_count() or 4)


class SepsisLSTM(nn.Module):
    def __init__(self, n_channels=N_CHANNELS, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


def sample_rows(df, seed=SEED):
    """All positives + NEG_MULTIPLE x random negatives (documented subsample)."""
    rng = np.random.RandomState(seed)
    pos = df[df[LABEL] == 1]
    neg = df[df[LABEL] == 0]
    neg_sel = neg.iloc[rng.choice(len(neg), len(pos) * NEG_MULTIPLE, replace=False)]
    return pd.concat([pos, neg_sel]).sample(frac=1, random_state=seed).reset_index(drop=True)


def build_sequences(rows, by_pid, stats):
    """Trailing LOOKBACK hours of standardized signals + missingness masks.

    Short histories are left-padded with zeros (mask 0) — the net sees
    explicitly what is padding vs measurement.
    """
    mean, std = stats["mean"], stats["std"]
    X = np.zeros((len(rows), LOOKBACK, 2 * len(SIGNALS)), dtype=np.float32)
    for i, (pid, t) in enumerate(zip(rows["pid"], rows["ICULOS"])):
        hist = by_pid[pid]
        hrs = hist["ICULOS"].to_numpy()
        j = int(np.searchsorted(hrs, t, side="right"))  # rows with hour <= t
        window = hist.iloc[max(0, j - LOOKBACK) : j]
        vals = window[SIGNALS].to_numpy(dtype=np.float32)
        mask = (~np.isnan(vals)).astype(np.float32)
        vals = np.where(mask, (vals - mean) / std, 0.0).astype(np.float32)
        seq = np.concatenate([vals, mask], axis=1)
        X[i, -len(seq):] = seq
    return X


def dataset_for(df, by_pid, stats):
    X = build_sequences(df, by_pid, stats)
    y = df[LABEL].to_numpy(dtype=np.float32)
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def main():
    df = pd.read_parquet(WINDOWS_PATH)
    df = exclude_post_onset(df)
    train, test = split_by_patient(df)
    train_s, test_s = sample_rows(train), sample_rows(test)
    by_pid = {pid: g.sort_values("ICULOS") for pid, g in df.groupby("pid")}

    obs = train[SIGNALS].to_numpy(dtype=np.float64)
    stats = {
        "mean": np.nanmean(obs, axis=0).astype(np.float32),
        "std": np.nanstd(obs, axis=0).astype(np.float32) + 1e-6,
    }

    train_ds = dataset_for(train_s, by_pid, stats)
    test_ds = dataset_for(test_s, by_pid, stats)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_X, test_y = test_ds.tensors

    pos_weight = torch.tensor([(train_s[LABEL] == 0).sum() / (train_s[LABEL] == 1).sum()])
    model = SepsisLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    for ep in range(1, EPOCHS + 1):
        tot = 0.0
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        print(f"  epoch {ep}/{EPOCHS} loss={tot / len(train_ds):.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(test_X)).numpy()
    y_true = test_y.numpy().astype(int)
    lstm_m = {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
    }

    # Like-for-like: score the saved LightGBM on the SAME sampled rows.
    lgbm = joblib.load(LGBM_PATH)
    cols = feature_columns(train)
    lgbm_proba = lgbm.predict_proba(test_s[cols])[:, 1]
    lgbm_m = metrics_at(y_true, lgbm_proba)

    torch.save(
        {"state_dict": model.state_dict(), "lookback": LOOKBACK, "signals": SIGNALS},
        LSTM_PATH,
    )
    METRICS_PATH.write_text(json.dumps({"lstm_sampled": lstm_m}, indent=2))
    COMPARE_PATH.write_text(
        json.dumps({"lstm_sampled": lstm_m, "lgbm_sampled": lgbm_m}, indent=2)
    )
    print(json.dumps({"lstm_sampled": lstm_m, "lgbm_sampled": lgbm_m}, indent=2))
    print(f"saved -> {LSTM_PATH}")


if __name__ == "__main__":
    main()
