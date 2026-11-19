"""Learning curve: is the model data-limited? Fit on a fraction of train prompts, score on val.

    python experiments/learning_curve.py --model svm --C 0.3 --min-prompts 3 \\
        --hypothesis "..." --predicted "..."

Each fraction < 1 is drawn several times (--repeats, different prompt subsets) and averaged,
so the curve isn't one lucky draw. TF-IDF is refit on each subset only (train split only).
Still climbing at 100% -> more data should help; flat -> features/model are the limit.
Output: runs/learning_curve/<timestamp>_<model>/ (run.log, summary.json).
"""
import argparse
import json

import numpy as np
import pandas as pd

from ntllm.data.scoring import score
from ntllm.data.splits import load_split
from ntllm.features.tfidf import TfidfFeatures
from ntllm.models.linear import make_model
from ntllm.runs import RunLog


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", choices=["nb", "svm"], default="svm")
    ap.add_argument("--C", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--min-prompts", type=int, default=3)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--repeats", type=int, default=5, help="prompt subsets per fraction < 1")
    ap.add_argument("--hypothesis", default=None)
    ap.add_argument("--predicted", default=None)
    a = ap.parse_args()

    run = RunLog("learning_curve", a.model, vars(a))
    run.header(a.hypothesis, a.predicted)
    train, val = load_split("train"), load_split("val")
    prompts = np.array(sorted(train.prompt_id.unique()))
    rng = np.random.default_rng(0)
    run.log(f"val: {len(val)} rows / {val.prompt_id.nunique()} prompts")

    summary = []
    for frac in a.fractions:
        n = max(2, round(frac * len(prompts)))
        reps = 1 if n == len(prompts) else a.repeats
        baccs = []
        for r in range(reps):
            sub = train[train.prompt_id.isin(rng.choice(prompts, n, replace=False))]
            # min_prompts is a count, so scale it with the subset to keep the filter comparable
            mp = max(1, round(a.min_prompts * n / len(prompts)))
            feats = TfidfFeatures(analyzer="both", min_prompts=mp).fit(sub)  # same features as train.py
            feats.assert_disjoint(val)
            kw = dict(C=a.C) if a.model == "svm" else dict(alpha=a.alpha)
            model = make_model(a.model, **kw).fit(feats.transform(sub.text), sub.model)
            baccs.append(score(val.model, model.predict(feats.transform(val.text)))["balanced_accuracy"])
        m, s = float(np.mean(baccs)), float(np.std(baccs))
        summary.append({"fraction": frac, "prompts": n, "rows": int(len(sub)), "min_prompts": mp,
                        "val_mean": m, "val_std": s, "repeats": baccs})
        run.log(f"{frac:>5.0%}  {n:>3} prompts ({len(sub):>4} rows, min_prompts={mp}): "
                f"val bal_acc {m:.4f} ± {s:.4f}  over {reps} subset(s)")

    gains = [(b["prompts"], b["val_mean"] - p["val_mean"]) for p, b in zip(summary, summary[1:])]
    run.log("gain per step: " + "  ".join(f"->{n}p {g:+.4f}" for n, g in gains))
    run.log(f"ACTUAL: val bal_acc at 100% = {summary[-1]['val_mean']:.4f}, last-step gain "
            f"{gains[-1][1]:+.4f} (predicted: {a.predicted or 'n/a'})")
    (run.dir / "summary.json").write_text(json.dumps({"curve": summary}, indent=2))


if __name__ == "__main__":
    main()
