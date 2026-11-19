"""Scoring: balanced accuracy (the competition metric), per-label recall, confusion matrix.

Balanced accuracy = mean of per-label recall over the labels present in y_true.
Worked example (DESIGN.md addenda): 1,000 GPT right + 50 Claude wrong -> accuracy 0.95,
balanced accuracy 0.5.
"""
import json

import numpy as np
import pandas as pd

from ntllm.data.load import LABELS


def confusion(y_true, y_pred, labels=LABELS) -> pd.DataFrame:
    """Rows = true label, columns = predicted label. Unknown predictions raise."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    assert len(y_true) == len(y_pred), "y_true and y_pred differ in length"
    idx = {l: i for i, l in enumerate(labels)}
    bad = set(y_pred) - set(idx)
    assert not bad, f"predictions not in labels: {bad}"
    m = np.zeros((len(labels), len(labels)), dtype=int)
    np.add.at(m, ([idx[t] for t in y_true], [idx[p] for p in y_pred]), 1)
    return pd.DataFrame(m, index=pd.Index(labels, name="true"),
                        columns=pd.Index(labels, name="pred"))


def score(y_true, y_pred, labels=LABELS) -> dict:
    cm = confusion(y_true, y_pred, labels)
    support = cm.sum(axis=1)
    present = support > 0
    recall = (np.diag(cm.values) / support.where(present)).rename("recall")
    predicted = cm.sum(axis=0)
    # Precision of a label never predicted counts as 0 (it found none of its rows).
    precision = pd.Series(np.diag(cm.values) / predicted.replace(0, np.nan), index=cm.index).fillna(0.0)
    return {
        "balanced_accuracy": float(recall[present].mean()),
        "macro_precision": float(precision[present].mean()),
        "accuracy": float(np.trace(cm.values) / cm.values.sum()),
        "per_label_recall": recall[present].to_dict(),
        "support": support[present].astype(int).to_dict(),
        "confusion": cm,
    }


def report(result: dict, title: str = "") -> str:
    """Human-readable summary: headline scores, per-label recall (worst first), top confusions."""
    lines = [f"== {title}" if title else "=="]
    lines.append(f"balanced_accuracy {result['balanced_accuracy']:.4f}   "
                 f"macro_precision {result['macro_precision']:.4f}   "
                 f"accuracy {result['accuracy']:.4f}")
    lines.append("per-label recall (worst first):")
    for label, r in sorted(result["per_label_recall"].items(), key=lambda kv: kv[1]):
        lines.append(f"  {label:<24} {r:.3f}  (n={result['support'][label]})")
    off = result["confusion"].to_numpy(copy=True)  # pandas 3: .values is read-only
    np.fill_diagonal(off, 0)
    pairs = pd.DataFrame(off, index=result["confusion"].index,
                         columns=result["confusion"].columns).stack()
    pairs = pairs[pairs > 0].sort_values(ascending=False).head(10)
    if len(pairs):
        lines.append("top confusions (true -> pred):")
        for (t, p), n in pairs.items():
            lines.append(f"  {t:<24} -> {p:<24} {n}")
    return "\n".join(lines)


def save(result: dict, path) -> None:
    """Write a score to JSON (confusion matrix as nested lists) so Claude can read results."""
    out = {k: v for k, v in result.items() if k != "confusion"}
    out["labels"] = list(result["confusion"].index)
    out["confusion"] = result["confusion"].values.tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
