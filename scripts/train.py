"""Train TF-IDF + Naive Bayes or TF-IDF + linear SVM (Chunk 3).

    python scripts/train.py --model svm --eval cv --C 0.1 0.3 1 3 10      # tune: 4-fold CV in train
    python scripts/train.py --model nb  --eval cv --alpha 0.01 0.03 0.1 0.3 1
    python scripts/train.py --model svm --eval cv --min-prompts 1 3       # ablation 4 (filter off/on)
    python scripts/train.py --model svm --eval val --C 1                  # fit on train, score val
    python scripts/train.py --model svm --eval test --C 1 \\
        --hypothesis "..." --predicted "..."                              # local test: once per finalist
    python scripts/train.py --model svm --eval cv --limit 30              # smoke test (30 train prompts)

Every combination of --min-prompts x (--alpha or --C) is one tuning run. In cv mode, each fold
refits TF-IDF on the other three train folds only, so the held-out fold never touches the
vocabulary; the fold-to-fold spread is the noise bar (open check 4).
val/test mode refits on the whole train split and saves per-class scores for blending.

Output: runs/<model>/<timestamp>_<eval>/ (run.log, config.json, summary.json, score_*.json,
predictions.csv, and for val/test: scores.npy, model.joblib).
"""
import argparse
import itertools
import json
import time

import joblib
import numpy as np
import pandas as pd

from ntllm.data.load import LABELS
from ntllm.data.scoring import report, save, score
from ntllm.data.splits import load_split
from ntllm.features.tfidf import TfidfFeatures
from ntllm.models.linear import make_model, perplexity, scores
from ntllm.runs import RunLog

N_FOLDS = 4


def feature_kwargs(a, min_prompts):
    return dict(analyzer=a.analyzer, word_ngram=a.word_ngram, char_ngram=a.char_ngram,
                min_df=a.min_df, max_features=a.max_features, lowercase=a.lowercase,
                min_prompts=min_prompts)


def model_kwargs(a, h):
    return dict(alpha=h, nb_kind=a.nb_kind) if a.model == "nb" else dict(C=h)


def fit_eval(a, fit_df, ev_df, min_prompts, hparams, run):
    """Fit TF-IDF on fit_df (train rows only), then one model per hparam. Yields results."""
    t = time.time()
    feats = TfidfFeatures(**feature_kwargs(a, min_prompts)).fit(fit_df)
    feats.assert_disjoint(ev_df)
    Xtr, Xev = feats.transform(fit_df["text"]), feats.transform(ev_df["text"])
    t_feat = time.time() - t
    for h in hparams:
        t = time.time()
        model = make_model(a.model, **model_kwargs(a, h)).fit(Xtr, fit_df["model"])
        t_fit = time.time() - t
        pred = model.predict(Xev)
        yield h, feats, model, Xev, pred, {"feat_s": t_feat, "fit_s": t_fit,
                                          "n_features": Xtr.shape[1], "stats": feats.stats_}


def run_cv(a, run, train, grid_mp, hparams, hname):
    summary = []
    for mp in grid_mp:
        oof = {h: pd.Series(index=train.index, dtype=object) for h in hparams}
        fold_bacc = {h: [] for h in hparams}
        fold_ppl = {h: [] for h in hparams}
        for k in range(N_FOLDS):
            fit_df, ev_df = train[train.fold != k], train[train.fold == k]
            for h, feats, model, Xev, pred, info in fit_eval(a, fit_df, ev_df, mp, hparams, run):
                oof[h].loc[ev_df.index] = pred
                r = score(ev_df["model"], pred)
                ppl = perplexity(model, Xev, ev_df["model"])
                fold_bacc[h].append(r["balanced_accuracy"])
                fold_ppl[h].append(ppl)
                run.log(f"  min_prompts={mp} {hname}={h} fold {k}: bal_acc {r['balanced_accuracy']:.4f} "
                        f"macro_prec {r['macro_precision']:.4f} ppl {_fmt(ppl)}  "
                        f"[{info['n_features']} feats, tfidf {info['feat_s']:.1f}s, fit {info['fit_s']:.1f}s]")
            if k == 0:
                run.log(f"  vocab (fold 0): {info['stats']}")
        for h in hparams:
            r = score(train["model"], oof[h].to_numpy())
            tag = f"mp{mp}_{hname}{h}"
            save(r, run.dir / f"score_cv_{tag}.json")
            m, s = float(np.mean(fold_bacc[h])), float(np.std(fold_bacc[h]))
            ppl = None if fold_ppl[h][0] is None else float(np.mean(fold_ppl[h]))
            summary.append({"min_prompts": mp, hname: h, "cv_mean": m, "cv_std": s,
                            "folds": fold_bacc[h], "perplexity": ppl,
                            "pooled_oof": r["balanced_accuracy"]})
            run.log(report(r, f"{a.model} min_prompts={mp} {hname}={h}: pooled out-of-fold, "
                              f"fold mean {m:.4f} ± {s:.4f}"))
            run.log()
    return summary


def run_holdout(a, run, train, ev, mp, h, hname):
    (h, feats, model, Xev, pred, info), = fit_eval(a, train, ev, mp, [h], run)
    run.log(f"vocab: {info['stats']}  (tfidf {info['feat_s']:.1f}s, fit {info['fit_s']:.1f}s)")
    run.log(f"most common dropped topic terms: {feats.dropped_examples()}")
    r = score(ev["model"], pred)
    ppl = perplexity(model, Xev, ev["model"])
    save(r, run.dir / f"score_{a.eval}.json")
    run.log(report(r, f"{a.model} min_prompts={mp} {hname}={h} on {a.eval}"))
    run.log(f"perplexity {_fmt(ppl)}")
    np.save(run.dir / "scores.npy", scores(model, Xev))  # rows = predictions.csv order, cols = LABELS
    pd.DataFrame({"id": ev.id, "prompt_id": ev.prompt_id, "model": ev.model, "pred": pred}) \
        .to_csv(run.dir / "predictions.csv", index=False)
    joblib.dump({"features": feats, "model": model, "labels": LABELS}, run.dir / "model.joblib")
    return [{"min_prompts": mp, hname: h, a.eval: r["balanced_accuracy"], "perplexity": ppl}]


def _fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", choices=["nb", "svm"], required=True)
    ap.add_argument("--eval", choices=["cv", "val", "test"], default="cv",
                    help="cv = 4 prompt-grouped folds inside train (tuning)")
    ap.add_argument("--C", type=float, nargs="+", default=[1.0], help="SVM regularization")
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.1], help="NB smoothing")
    ap.add_argument("--nb-kind", choices=["complement", "multinomial"], default="complement")
    ap.add_argument("--min-prompts", type=int, nargs="+", default=[3],
                    help="topic-word filter: keep terms seen in >= N train prompts (1 = off)")
    ap.add_argument("--analyzer", choices=["word", "char", "both"], default="both")
    ap.add_argument("--word-ngram", type=int, nargs=2, default=[1, 2])
    ap.add_argument("--char-ngram", type=int, nargs=2, default=[2, 5])
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--max-features", type=int, default=None, help="per analyzer")
    ap.add_argument("--lowercase", action="store_true", help="default keeps case (style signal)")
    ap.add_argument("--limit", type=int, default=None, help="use only N train prompts (smoke)")
    ap.add_argument("--hypothesis", default=None)
    ap.add_argument("--predicted", default=None)
    a = ap.parse_args()

    hname = "alpha" if a.model == "nb" else "C"
    hparams = a.alpha if a.model == "nb" else a.C
    if a.eval == "test" and (len(hparams) > 1 or len(a.min_prompts) > 1):
        ap.error("local test is scored once: tune with --eval cv, then pass one config")

    name = a.eval + (f"_smoke{a.limit}" if a.limit else "")
    run = RunLog(a.model, name, vars(a))
    run.header(a.hypothesis, a.predicted)

    train = load_split("train")
    if a.limit:
        keep = pd.Series(train.prompt_id.unique()).sample(n=a.limit, random_state=0)
        train = train[train.prompt_id.isin(keep)].reset_index(drop=True)
    run.log(f"train: {len(train)} rows / {train.prompt_id.nunique()} prompts; "
            f"{len(hparams) * len(a.min_prompts)} config(s)")

    if a.eval == "cv":
        summary = run_cv(a, run, train, a.min_prompts, hparams, hname)
        best = max(summary, key=lambda s: s["cv_mean"])
        run.log("cv balanced accuracy (fold mean ± std):")
        for s in summary:
            run.log(f"  min_prompts={s['min_prompts']:<3} {hname}={s[hname]:<8g} "
                    f"{s['cv_mean']:.4f} ± {s['cv_std']:.4f}  ppl {_fmt(s['perplexity'])}")
        run.log(f"ACTUAL: best min_prompts={best['min_prompts']} {hname}={best[hname]} "
                f"cv={best['cv_mean']:.4f} ± {best['cv_std']:.4f} (predicted: {a.predicted or 'n/a'})")
    else:
        ev = load_split(a.eval)
        run.log(f"eval on {a.eval}: {len(ev)} rows / {ev.prompt_id.nunique()} prompts")
        summary = run_holdout(a, run, train, ev, a.min_prompts[0], hparams[0], hname)
        run.log(f"ACTUAL: balanced_accuracy={summary[0][a.eval]:.4f} on {a.eval} "
                f"(predicted: {a.predicted or 'n/a'})")
    (run.dir / "summary.json").write_text(json.dumps(
        {"runs": summary, "seconds": run.elapsed()}, indent=2))
    run.log(f"total {run.elapsed():.1f}s")


if __name__ == "__main__":
    main()
