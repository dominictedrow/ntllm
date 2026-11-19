"""Fine-tune the encoder (Chunk 4). Run on the Windows RTX 5070; the Mac is for smoke tests.

    python scripts/train_encoder.py --eval cv --epochs 5                  # 4-fold CV in train: pick epochs/lr
    python scripts/train_encoder.py --eval cv --folds 0 --lr 3e-5         # one fold, quick check
    python scripts/train_encoder.py --eval cv --family                    # Experiment 2: + gzip family prefix
    python scripts/train_encoder.py --eval val --epochs 4                 # fit on all train, score val, save weights
    python scripts/train_encoder.py --eval test --epochs 4 \\
        --hypothesis "..." --predicted "..."                              # local test: once per finalist
    python scripts/train_encoder.py --eval cv --folds 0 --limit 8 --max-steps 5 --epochs 1   # smoke

cv: each fold trains on the other three train folds and is scored after every epoch, so
the per-epoch fold mean tells you how many epochs to use. Val/test never pick epochs.
val/test: trains on the whole train split for --epochs, then scores once; weights go to
checkpoints/encoder/<run>/final (export with scripts/export_encoder.py).

Output: runs/encoder/<timestamp>_<eval>/ (run.log, config.json, history.json, summary.json,
score_*.json, and for val/test predictions.csv + scores.npy [logits, LABELS order]).
"""
import argparse
import json

import numpy as np
import pandas as pd

from ntllm.data.load import LABELS
from ntllm.data.scoring import report, save, score
from ntllm.data.splits import load_split
from ntllm.features.family import FAMILY, GzipFamily, add_prefix
from ntllm.models.encoder import EncoderClassifier, evaluate
from ntllm.paths import CHECKPOINTS
from ntllm.runs import RunLog

N_FOLDS = 4


def family_texts(fit_df, ev_df, a, run):
    """Prefix texts with gzip-guessed family. Fit rows: leave-prompt-out; eval rows: full index."""
    if not a.family:
        return fit_df.text.tolist(), ev_df.text.tolist()
    t = run.elapsed()
    gf = GzipFamily(k=a.family_k).fit(fit_df)
    f_fit, f_ev = gf.predict_fit_rows(), gf.predict(ev_df)
    acc_fit = np.mean(f_fit == fit_df.model.map(FAMILY).to_numpy())
    acc_ev = np.mean(f_ev == ev_df.model.map(FAMILY).to_numpy())
    run.log(f"gzip family k={a.family_k}: accuracy {acc_fit:.3f} on fit rows (leave-prompt-out), "
            f"{acc_ev:.3f} on eval rows  [{run.elapsed() - t:.0f}s]")
    return add_prefix(fit_df.text, f_fit), add_prefix(ev_df.text, f_ev)


def make(a, run):
    return EncoderClassifier(backbone=a.backbone, max_len=a.max_len, lr=a.lr, epochs=a.epochs,
                             batch_size=a.batch_size, grad_accum=a.grad_accum, warmup=a.warmup,
                             weight_decay=a.weight_decay, seed=a.seed, device=a.device,
                             bf16=not a.no_bf16, log=run.log)


def run_cv(a, run, train):
    per_fold = {}
    for k in a.folds:
        fit_df, ev_df = train[train.fold != k], train[train.fold == k]
        assert not set(fit_df.prompt_id) & set(ev_df.prompt_id)
        run.log(f"--- fold {k}: fit {len(fit_df)} rows / eval {len(ev_df)} rows")
        x_fit, x_ev = family_texts(fit_df, ev_df, a, run)
        enc = make(a, run)
        if k == a.folds[0]:
            run.log(f"device {enc.device}  bf16 {enc.amp}")
        hist = enc.fit(x_fit, fit_df.model.tolist(), x_ev, ev_df.model.tolist(),
                       max_steps=a.max_steps)
        per_fold[k] = [{kk: v for kk, v in h.items() if not kk.startswith("_")} for h in hist]
        last = hist[-1]["_score"]
        run.log(report(last, f"fold {k}, last epoch"))
        del enc
    # Per-epoch mean over folds -> choose epochs.
    n_ep = min(len(h) for h in per_fold.values())
    rows = []
    for e in range(n_ep):
        b = [per_fold[k][e]["balanced_accuracy"] for k in per_fold]
        p = [per_fold[k][e]["perplexity"] for k in per_fold]
        s = [per_fold[k][e]["epoch_s"] for k in per_fold]
        rows.append({"epoch": e + 1, "cv_mean": float(np.mean(b)), "cv_std": float(np.std(b)),
                     "perplexity": float(np.mean(p)), "epoch_s": float(np.mean(s))})
    run.log("cv by epoch (fold mean ± std):")
    for r in rows:
        run.log(f"  epoch {r['epoch']}: bal_acc {r['cv_mean']:.4f} ± {r['cv_std']:.4f}  "
                f"ppl {r['perplexity']:.3f}  {r['epoch_s']:.0f}s/epoch")
    best = max(rows, key=lambda r: r["cv_mean"])
    run.log(f"ACTUAL: best epoch {best['epoch']} cv={best['cv_mean']:.4f} ± {best['cv_std']:.4f} "
            f"over folds {a.folds} (predicted: {a.predicted or 'n/a'})")
    (run.dir / "history.json").write_text(json.dumps(per_fold, indent=2))
    return {"by_epoch": rows, "best": best}


def run_holdout(a, run, train, ev):
    x_fit, x_ev = family_texts(train, ev, a, run)
    enc = make(a, run)
    run.log(f"device {enc.device}  bf16 {enc.amp}")
    hist = enc.fit(x_fit, train.model.tolist(), max_steps=a.max_steps)
    logits = enc.predict_logits(x_ev)
    m = evaluate(logits, ev.model.tolist())
    r = m.pop("_score")
    save(r, run.dir / f"score_{a.eval}.json")
    run.log(report(r, f"encoder {a.backbone} on {a.eval}"))
    run.log(f"perplexity {m['perplexity']:.3f}")
    np.save(run.dir / "scores.npy", logits)
    pd.DataFrame({"id": ev.id, "prompt_id": ev.prompt_id, "model": ev.model,
                  "pred": np.array(LABELS)[logits.argmax(1)]}).to_csv(run.dir / "predictions.csv", index=False)
    ckpt = CHECKPOINTS / "encoder" / run.dir.name / "final"
    enc.save(ckpt, extra={"family": a.family, "family_k": a.family_k, "trained_on": "train split v1",
                          a.eval: m["balanced_accuracy"]})
    run.log(f"weights: {ckpt}")
    (run.dir / "history.json").write_text(json.dumps(hist, indent=2))
    run.log(f"ACTUAL: balanced_accuracy={m['balanced_accuracy']:.4f} on {a.eval} "
            f"(predicted: {a.predicted or 'n/a'})")
    return {a.eval: m["balanced_accuracy"], "perplexity": m["perplexity"], "checkpoint": str(ckpt)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--eval", choices=["cv", "val", "test"], default="cv")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)), help="cv folds to run")
    ap.add_argument("--backbone", default="microsoft/deberta-v3-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=384, help="longest train passage is 351 tokens")
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--family", action="store_true", help="Experiment 2: prefix gzip-guessed family")
    ap.add_argument("--family-k", type=int, default=7)
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="use only N train prompts (smoke)")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps (smoke)")
    ap.add_argument("--hypothesis", default=None)
    ap.add_argument("--predicted", default=None)
    a = ap.parse_args()

    name = a.eval + ("_family" if a.family else "") + (f"_smoke{a.limit}" if a.limit else "")
    run = RunLog("encoder", name, vars(a))
    run.header(a.hypothesis, a.predicted)
    train = load_split("train")
    if a.limit:
        keep = pd.Series(train.prompt_id.unique()).sample(n=a.limit, random_state=0)
        train = train[train.prompt_id.isin(keep)].reset_index(drop=True)
    run.log(f"train: {len(train)} rows / {train.prompt_id.nunique()} prompts; backbone {a.backbone}")

    if a.eval == "cv":
        summary = run_cv(a, run, train)
    else:
        ev = load_split(a.eval)
        if a.limit:
            ev = ev[ev.prompt_id.isin(ev.prompt_id.unique()[:max(1, a.limit // 3)])].reset_index(drop=True)
        run.log(f"eval on {a.eval}: {len(ev)} rows / {ev.prompt_id.nunique()} prompts")
        summary = run_holdout(a, run, train, ev)
    summary["seconds"] = run.elapsed()
    (run.dir / "summary.json").write_text(json.dumps(summary, indent=2))
    run.log(f"total {run.elapsed():.1f}s")


if __name__ == "__main__":
    main()
