"""Make the prompt-grouped train / val / local-test splits once.

    python scripts/make_splits.py                      # defaults: 60/20/20 by prompt, 4 folds
    python scripts/make_splits.py --val-frac 0.25 --test-frac 0.15 --version v2

Writes data/splits/<version>/assignments.csv and manifest.json. Refuses to overwrite.
"""
import argparse

from ntllm.data.load import load_train
from ntllm.data.splits import DEFAULT_VERSION, make_splits, save_splits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--val-frac", type=float, default=0.2, help="share of prompts in val")
    ap.add_argument("--test-frac", type=float, default=0.2, help="share of prompts in local test")
    ap.add_argument("--folds", type=int, default=4, help="prompt-grouped folds inside train and val")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--version", default=DEFAULT_VERSION)
    ap.add_argument("--force", action="store_true", help="overwrite an existing version")
    a = ap.parse_args()

    config = {"val_frac": a.val_frac, "test_frac": a.test_frac, "folds": a.folds, "seed": a.seed}
    assign = make_splits(load_train(), a.val_frac, a.test_frac, a.folds, a.seed)
    m = save_splits(assign, config, a.version, a.force)
    print(f"splits {a.version} written  config={config}")
    for name, s in m["summary"].items():
        print(f"  {name:<5} prompts={s['prompts']:<4} rows={s['rows']:<5} "
              f"prompts/fold={s['prompts_per_fold']}")


if __name__ == "__main__":
    main()
