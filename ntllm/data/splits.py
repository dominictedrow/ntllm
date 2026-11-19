"""Prompt-grouped train / val / local-test splits, made once and reused.

Design (docs/DESIGN.md, Validation): three splits of train.csv, all separated by prompt.
Base models fit on `train`; the meta-model fits on base-model predictions on `val`;
`test` (local test) is scored once per finalist. Kaggle's test.csv is only for submissions.

Each row also gets a `fold` (0..n_folds-1) inside train and inside val, grouped by prompt:
train folds are for tuning individual models (4-fold k-fold val in the design); val folds
let meta-model selection happen inside val so local test is touched once (open check 3).
Local-test rows get fold = -1.
"""
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ntllm.data.load import load_train
from ntllm.paths import SPLITS

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_VERSION = "v1"


def make_splits(df: pd.DataFrame, val_frac: float, test_frac: float,
                n_folds: int, seed: int) -> pd.DataFrame:
    """Return a table with columns id, prompt_id, split, fold (one row per train.csv row)."""
    prompts = np.array(sorted(df["prompt_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(prompts)

    n = len(prompts)
    n_test = round(n * test_frac)
    n_val = round(n * val_frac)
    groups = {
        "test": prompts[:n_test],
        "val": prompts[n_test:n_test + n_val],
        "train": prompts[n_test + n_val:],
    }

    prompt_split, prompt_fold = {}, {}
    for name, ps in groups.items():
        for i, p in enumerate(ps):
            prompt_split[p] = name
            # Prompts are already shuffled, so round-robin gives random, near-equal folds.
            prompt_fold[p] = -1 if name == "test" else i % n_folds

    out = df[["id", "prompt_id"]].copy()
    out["split"] = out["prompt_id"].map(prompt_split)
    out["fold"] = out["prompt_id"].map(prompt_fold).astype(int)
    assert_prompt_disjoint(out)
    return out


def assert_prompt_disjoint(assign: pd.DataFrame) -> None:
    """No prompt may appear in more than one split, or in more than one fold of a split."""
    n_splits = assign.groupby("prompt_id")["split"].nunique()
    leaked = n_splits[n_splits > 1]
    assert leaked.empty, f"prompts in more than one split: {list(leaked.index)}"
    n_folds = assign.groupby("prompt_id")["fold"].nunique()
    leaked = n_folds[n_folds > 1]
    assert leaked.empty, f"prompts in more than one fold: {list(leaked.index)}"
    assert assign["id"].is_unique
    assert set(assign["split"]) <= set(SPLIT_NAMES)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_splits(assign: pd.DataFrame, config: dict, version: str = DEFAULT_VERSION,
                force: bool = False) -> dict:
    """Write data/splits/<version>/assignments.csv + manifest.json. Refuses to overwrite."""
    out_dir = SPLITS / version
    path = out_dir / "assignments.csv"
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; splits are made once. Use a new --version "
                              "or --force (this changes every downstream score).")
    out_dir.mkdir(parents=True, exist_ok=True)
    assign.sort_values("id").to_csv(path, index=False)

    from ntllm.paths import RAW
    manifest = {
        "version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "train_csv_sha256": _sha256(RAW / "train.csv"),
        "assignments_sha256": _sha256(path),
        "summary": summarize(assign),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def summarize(assign: pd.DataFrame) -> dict:
    s = {}
    for name in SPLIT_NAMES:
        part = assign[assign["split"] == name]
        s[name] = {
            "prompts": int(part["prompt_id"].nunique()),
            "rows": int(len(part)),
            "prompts_per_fold": {int(k): int(v) for k, v in
                                 part.groupby("fold")["prompt_id"].nunique().items()},
        }
    return s


def load_assignments(version: str = DEFAULT_VERSION) -> pd.DataFrame:
    path = SPLITS / version / "assignments.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python scripts/make_splits.py")
    manifest = json.loads((SPLITS / version / "manifest.json").read_text())
    if _sha256(path) != manifest["assignments_sha256"]:
        raise RuntimeError(f"{path} changed since it was made; splits must not be edited.")
    from ntllm.paths import RAW
    if _sha256(RAW / "train.csv") != manifest["train_csv_sha256"]:
        raise RuntimeError("train.csv differs from the one the splits were made from.")
    assign = pd.read_csv(path)
    assert_prompt_disjoint(assign)
    return assign


def load_split(name: str, version: str = DEFAULT_VERSION, fold=None) -> pd.DataFrame:
    """Rows of train.csv for one split (train / val / test), with split + fold columns.

    fold=None returns the whole split; fold=k returns only that fold.
    """
    if name not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {name!r}")
    df = load_train().merge(load_assignments(version)[["id", "split", "fold"]],
                            on="id", how="left", validate="one_to_one")
    assert df["split"].notna().all(), "train.csv rows missing from the split file"
    part = df[df["split"] == name]
    if fold is not None:
        part = part[part["fold"] == fold]
    return part.reset_index(drop=True)
