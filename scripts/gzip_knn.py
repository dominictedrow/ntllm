"""Gzip + kNN baseline: fit on the train split, score on val or local test.

    python scripts/gzip_knn.py --eval val --k 1 3 5 7 9          # pick k on val
    python scripts/gzip_knn.py --eval test --k 3 \\
        --hypothesis "gzip+kNN finds model style" --predicted "bal acc ~0.30"
    python scripts/gzip_knn.py --eval val --limit 40              # smoke test

Distances are computed once; several k values are scored from the same matrix.
Local test takes exactly one k (it is scored once per finalist).
Output: runs/gzip_knn/<timestamp>_<eval>/ (run.log, config.json, score_k*.json,
predictions.csv, distances.npy, class_min_dist.npy).
"""
import argparse
import json

import numpy as np
import pandas as pd

from ntllm.data.scoring import report, save, score
from ntllm.data.splits import load_split
from ntllm.models.gzip_knn import GzipKNN
from ntllm.runs import RunLog


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--eval", choices=["val", "test"], default="val", help="split to score on")
    ap.add_argument("--k", type=int, nargs="+", default=[1], help="neighbors (several allowed on val)")
    ap.add_argument("--level", type=int, default=9, help="gzip compression level 1-9")
    ap.add_argument("--jobs", type=int, default=None, help="worker processes (default: all cores)")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N eval rows (smoke)")
    ap.add_argument("--hypothesis", default=None)
    ap.add_argument("--predicted", default=None)
    a = ap.parse_args()
    if a.eval == "test" and len(a.k) > 1:
        ap.error("local test is scored once: pick k on val, then pass a single --k")

    name = a.eval + (f"_smoke{a.limit}" if a.limit else "")
    run = RunLog("gzip_knn", name, vars(a))
    run.header(a.hypothesis, a.predicted)

    train, ev = load_split("train"), load_split(a.eval)
    if a.limit:
        ev = ev.sample(n=min(a.limit, len(ev)), random_state=0).reset_index(drop=True)
    run.log(f"fit on train: {len(train)} rows / {train.prompt_id.nunique()} prompts; "
            f"eval on {a.eval}: {len(ev)} rows")

    model = GzipKNN(k=a.k[0], level=a.level, n_jobs=a.jobs).fit(train.text, train.model)
    t = run.elapsed()
    D = model.distances(ev.text.tolist())
    secs = run.elapsed() - t
    run.log(f"distances: {D.shape} in {secs:.1f}s  ({secs / len(ev) * 1000:.1f} ms/row, "
            f"{model.n_jobs} workers)")

    np.save(run.dir / "distances.npy", D)
    np.save(run.dir / "class_min_dist.npy", model.class_min_distance(D))
    preds = pd.DataFrame({"id": ev.id, "prompt_id": ev.prompt_id, "model": ev.model})
    summary = {}
    for k in a.k:
        p = model.predict_from_distances(D, k)
        preds[f"pred_k{k}"] = p
        r = score(ev.model, p)
        save(r, run.dir / f"score_k{k}.json")
        summary[k] = r["balanced_accuracy"]
        run.log(report(r, f"gzip+kNN k={k} level={a.level} on {a.eval}"))
        run.log()
    preds.to_csv(run.dir / "predictions.csv", index=False)

    best = max(summary, key=summary.get)
    run.log("balanced accuracy by k: " + "  ".join(f"k={k}: {v:.4f}" for k, v in summary.items()))
    run.log(f"ACTUAL: best k={best} balanced_accuracy={summary[best]:.4f} on {a.eval} "
            f"(predicted: {a.predicted or 'n/a'})")
    (run.dir / "summary.json").write_text(json.dumps(
        {"balanced_accuracy_by_k": summary, "seconds": run.elapsed()}, indent=2))
    run.log(f"total {run.elapsed():.1f}s")


if __name__ == "__main__":
    main()
