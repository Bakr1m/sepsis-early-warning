"""Hermetic tests for time-respecting evaluation helpers (no data/)."""
import numpy as np

from src.evaluate import assign_folds, precision_recall_at, threshold_for_recall


def test_assign_folds_exclusive_covering_stratified():
    pids = [f"p{i}" for i in range(12)]
    flags = [1, 1, 1, 1] + [0] * 8  # 4 septic, 8 clean
    f = assign_folds(pids, flags, n_folds=4, seed=0)
    assert set(f) == set(pids)  # every patient assigned exactly once
    per_fold = {}
    for p, k in f.items():
        per_fold.setdefault(k, []).append(p)
    assert len(per_fold) == 4
    for members in per_fold.values():  # each fold has both classes
        assert sum(flags[pids.index(p)] for p in members) == 1
        assert len(members) == 3


def test_precision_recall_at_exact():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    m = precision_recall_at(y, p, 0.35)
    assert m["precision"] == round(2 / 3, 4)
    assert m["recall"] == 1.0
    assert m["alert_rate"] == 0.75


def test_threshold_for_recall_meets_target():
    rng = np.random.RandomState(0)
    y = (rng.rand(500) < 0.1).astype(int)
    p = rng.rand(500)
    t = threshold_for_recall(y, p, target=0.8)
    assert ((p >= t) & (y == 1)).sum() / y.sum() >= 0.8


def test_threshold_for_recall_is_highest_valid():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    assert threshold_for_recall(y, p, target=0.5) == 0.8
    assert threshold_for_recall(y, p, target=1.0) == 0.35
