import numpy as np
import pandas as pd
import pytest

from ntllm.data.load import LABELS, load_train
from ntllm.data.scoring import score
from ntllm.data.splits import (SPLIT_NAMES, assert_prompt_disjoint, load_assignments,
                               load_split, make_splits)
from ntllm.paths import SPLITS


def test_make_splits_prompt_disjoint_and_deterministic():
    df = load_train()
    a = make_splits(df, 0.2, 0.2, 4, seed=0)
    b = make_splits(df, 0.2, 0.2, 4, seed=0)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == len(df)
    assert a.groupby("prompt_id")["split"].nunique().max() == 1
    assert (a.loc[a["split"] == "test", "fold"] == -1).all()
    assert set(a.loc[a["split"] != "test", "fold"]) == {0, 1, 2, 3}


def test_disjoint_check_catches_leak():
    bad = pd.DataFrame({"id": ["a", "b"], "prompt_id": ["p1", "p1"],
                        "split": ["train", "val"], "fold": [0, 0]})
    with pytest.raises(AssertionError):
        assert_prompt_disjoint(bad)


@pytest.mark.skipif(not (SPLITS / "v1" / "assignments.csv").exists(), reason="no saved splits")
def test_saved_splits_load_and_cover_train():
    assign = load_assignments()
    assert len(assign) == len(load_train())
    seen = set()
    for name in SPLIT_NAMES:
        p = set(load_split(name)["prompt_id"])
        assert not (p & seen)
        seen |= p


def test_score_worked_example():
    # DESIGN.md: 1,000 GPT right + 50 Claude wrong -> accuracy 0.95, balanced accuracy 0.5
    y_true = ["gpt-5.5"] * 1000 + ["claude-opus-5"] * 50
    y_pred = ["gpt-5.5"] * 1050
    r = score(y_true, y_pred)
    assert r["accuracy"] == pytest.approx(1000 / 1050)
    assert r["balanced_accuracy"] == pytest.approx(0.5)
    assert r["per_label_recall"]["claude-opus-5"] == 0
    assert r["confusion"].loc["claude-opus-5", "gpt-5.5"] == 50


def test_score_matches_sklearn():
    from sklearn.metrics import balanced_accuracy_score
    rng = np.random.default_rng(0)
    y_true = rng.choice(LABELS, 500)
    y_pred = np.where(rng.random(500) < 0.6, y_true, rng.choice(LABELS, 500))
    assert score(y_true, y_pred)["balanced_accuracy"] == pytest.approx(
        balanced_accuracy_score(y_true, y_pred))
