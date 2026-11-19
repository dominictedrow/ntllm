"""Blending (Chunk 5): base models fit on train predict val; a meta-model learns from those val
predictions and is scored once on local test. Plain averaging is the comparison.

Base models give scores on different scales (SVM margins, NB log-probs, encoder logits, gzip
distances), so each model's row is z-scored across the 18 labels before averaging or stacking.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ntllm.data.load import LABELS


def load_scores(run_dir) -> pd.DataFrame:
    """Per-class scores from a base-model val/test run: index = id, columns = LABELS, higher = likelier.

    train.py / train_encoder.py runs save scores.npy; gzip_knn.py runs save class_min_dist.npy
    (distance, lower = likelier), which is negated.
    """
    d = Path(run_dir)
    ids = pd.read_csv(d / "predictions.csv")["id"]
    s = np.load(d / "scores.npy") if (d / "scores.npy").exists() else -np.load(d / "class_min_dist.npy")
    assert s.shape == (len(ids), len(LABELS)), f"{d}: scores {s.shape}, expected {(len(ids), len(LABELS))}"
    return pd.DataFrame(s, index=ids.values, columns=LABELS)


def standardize(s) -> np.ndarray:
    """z-score each row across labels, so every base model votes on the same scale."""
    s = np.asarray(s, dtype=float)
    return (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-9)


def features(bases: dict, ids) -> np.ndarray:
    """Stack standardized scores of every base model (sorted by name) for the given row ids."""
    out = []
    for name in sorted(bases):
        df = bases[name]
        missing = set(ids) - set(df.index)
        extra = set(df.index) - set(ids)
        assert not missing and not extra, \
            f"base {name!r} rows don't match the split ({len(missing)} missing, {len(extra)} extra)"
        out.append(standardize(df.loc[list(ids)].values))
    return np.hstack(out)


def average_predict(X: np.ndarray) -> np.ndarray:
    """Plain averaging: mean of the standardized base scores, then argmax."""
    n = X.shape[1] // len(LABELS)
    return np.array(LABELS)[X.reshape(len(X), n, len(LABELS)).mean(axis=1).argmax(axis=1)]


def make_meta(C: float, seed: int = 0) -> LogisticRegression:
    # class_weight balanced: the metric is balanced accuracy.
    return LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed)
