"""Blend base-model predictions with a meta-model; plain averaging is the comparison (Chunk 5).

    # 1. pick the meta-model inside val (4 prompt-grouped folds), local test untouched:
    python scripts/blend.py --eval cv --C 0.01 0.1 1 10 \\
        --val svm=runs/svm/<ts>_val encoder=runs/encoder/<ts>_val gzip=runs/gzip_knn/<ts>_val
    # 2. score the chosen one once on local test (fit on all of val):
    python scripts/blend.py --eval test --C 0.1 --val svm=... encoder=... gzip=... \\
        --test svm=runs/svm/<ts>_test encoder=runs/encoder/<ts>_test gzip=runs/gzip_knn/<ts>_test \\
        --hypothesis "..." --predicted "..."

Base runs come from the base scripts' --eval val / --eval test modes (fit on train only).
Every base run is listed by name=run_dir; the same names must appear in --val and --test.

Output: runs/blend/<timestamp>_<eval>/ (run.log, config.json, summary.json, score_*.json,
and for test: predictions.csv).
"""
import argparse
import json

import numpy as np
import pandas as pd

from ntllm.data.load import LABELS
from ntllm.data.scoring import report, save, score
from ntllm.data.splits import load_split
from ntllm.models.blend import average_predict, features, load_scores, make_meta
from ntllm.runs import RunLog


def _bases(pairs):
    out = {}
    for p in pairs:
        name, _, path = p.partition("=")
        assert name and path, f"expected name=run_dir, got {p!r}"
        out[name] = load_scores(path)
    return out


def _alone(X, i):
    """Predictions of base model i alone (its block of standardized scores)."""
    return np.array(LABELS)[X[:, i * len(LABELS):(i + 1) * len(LABELS)].argmax(axis=1)]


def cv(a, run, val_bases):
    val = load_split("val")
    X, y, folds = features(val_bases, val.id), val.model.values, val.fold.values
    names = sorted(val_bases)
    methods = {f"meta C={c:g}": c for c in a.C} | {"average": None} | {f"{n} alone": n for n in names}
    per_fold = {m: [] for m in methods}
    for f in sorted(set(folds)):
        tr, ev = folds != f, folds == f
        for m, c in methods.items():
            if m == "average":
                p = average_predict(X[ev])
            elif m.endswith(" alone"):
                p = _alone(X[ev], names.index(c))
            else:
                p = make_meta(c).fit(X[tr], y[tr]).predict(X[ev])
            per_fold[m].append(score(y[ev], p)["balanced_accuracy"])
    run.log("val cv balanced accuracy (4 prompt-grouped folds inside val; fold mean ± std):")
    summary = {}
    for m, v in sorted(per_fold.items(), key=lambda kv: -np.mean(kv[1])):
        summary[m] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "folds": v}
        run.log(f"  {m:<24} {np.mean(v):.4f} ± {np.std(v):.4f}")
    best = max((m for m in methods if m.startswith("meta")), key=lambda m: summary[m]["mean"])
    run.log(f"ACTUAL: best {best} cv={summary[best]['mean']:.4f} vs average "
            f"{summary['average']['mean']:.4f} (predicted: {a.predicted or 'n/a'})")
    return summary


def test(a, run, val_bases, test_bases):
    assert len(a.C) == 1, "local test is scored once: pass exactly one --C (picked in --eval cv)"
    assert sorted(val_bases) == sorted(test_bases), \
        f"--val and --test must name the same base models: {sorted(val_bases)} vs {sorted(test_bases)}"
    val, te = load_split("val"), load_split("test")
    Xv, Xt, y = features(val_bases, val.id), features(test_bases, te.id), te.model.values
    meta = make_meta(a.C[0]).fit(Xv, val.model.values)
    preds = {"blend": meta.predict(Xt), "average": average_predict(Xt)}
    preds |= {f"{n}_alone": _alone(Xt, i) for i, n in enumerate(sorted(test_bases))}
    summary = {}
    for m, p in preds.items():
        r = score(y, p)
        save(r, run.dir / f"score_test_{m}.json")
        summary[m] = r["balanced_accuracy"]
        if m in ("blend", "average"):
            run.log(report(r, f"{m} on local test"))
            run.log()
    run.log("local test balanced accuracy: " + "  ".join(f"{m}: {v:.4f}" for m, v in summary.items()))
    pd.DataFrame({"id": te.id, "prompt_id": te.prompt_id, "model": y} |
                 {f"pred_{m}": p for m, p in preds.items()}).to_csv(run.dir / "predictions.csv", index=False)
    run.log(f"ACTUAL: blend={summary['blend']:.4f} average={summary['average']:.4f} on local test "
            f"(predicted: {a.predicted or 'n/a'})")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--eval", choices=["cv", "test"], default="cv")
    ap.add_argument("--val", nargs="+", required=True, help="name=run_dir of base --eval val runs")
    ap.add_argument("--test", nargs="+", help="name=run_dir of base --eval test runs (--eval test)")
    ap.add_argument("--C", type=float, nargs="+", default=[0.01, 0.1, 1, 10], help="meta-model regularization")
    ap.add_argument("--hypothesis")
    ap.add_argument("--predicted")
    a = ap.parse_args()
    assert a.eval == "cv" or a.test, "--eval test needs --test runs"

    run = RunLog("blend", a.eval, vars(a))
    run.header(a.hypothesis, a.predicted)
    val_bases = _bases(a.val)
    summary = cv(a, run, val_bases) if a.eval == "cv" else test(a, run, val_bases, _bases(a.test))
    (run.dir / "summary.json").write_text(json.dumps(summary | {"seconds": run.elapsed()}, indent=2))
    run.log(f"total {run.elapsed():.1f}s")


if __name__ == "__main__":
    main()
