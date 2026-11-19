import pandas as pd
import pytest

from ntllm.features.tfidf import TfidfFeatures, prompt_frequency
from ntllm.data.splits import load_split


def _df(split="train"):
    rows = []
    for p, topic in enumerate(["volcano lava", "quantum spin", "forest moss"]):
        for m, style in enumerate(["Certainly! Let us delve", "Here is the answer"]):
            rows.append({"id": f"{split}{p}{m}", "prompt_id": f"{split}p{p}", "split": split,
                         "text": f"{style}: {topic} {topic}.", "model": f"m{m}"})
    return pd.DataFrame(rows)


def test_topic_filter_drops_single_prompt_terms():
    f = TfidfFeatures(analyzer="word", word_ngram=(1, 1), min_df=1, min_prompts=2).fit(_df())
    names = set(f.feature_names())
    assert {"word:delve", "word:Certainly", "word:answer"} <= names     # style, in all 3 prompts
    assert not {"word:volcano", "word:quantum", "word:moss"} & names    # topic, in 1 prompt
    off = TfidfFeatures(analyzer="word", word_ngram=(1, 1), min_df=1, min_prompts=1).fit(_df())
    assert "word:volcano" in set(off.feature_names())


def test_refuses_to_fit_on_val_or_test():
    for split in ("val", "test"):
        with pytest.raises(AssertionError, match="train split only"):
            TfidfFeatures(analyzer="word").fit(pd.concat([_df(), _df(split)]))


def test_transform_rejects_shared_prompts():
    tr = _df()
    f = TfidfFeatures(analyzer="both", min_df=1, min_prompts=1).fit(tr)
    f.assert_disjoint(_df("val"))
    with pytest.raises(AssertionError):
        f.assert_disjoint(tr.iloc[:2])
    assert f.transform(_df("val").text).shape[0] == 6


def test_prompt_frequency_counts_prompts_not_rows():
    f = TfidfFeatures(analyzer="word", word_ngram=(1, 1), min_df=1, min_prompts=1)
    X = f.vecs["word"].fit_transform(_df().text)
    pf = dict(zip(f.vecs["word"].get_feature_names_out(), prompt_frequency(X, _df().prompt_id)))
    assert pf["volcano"] == 1 and pf["delve"] == 3


def test_real_train_split_fits():
    tr = load_split("train", fold=None)
    f = TfidfFeatures(analyzer="word", min_prompts=3).fit(tr)
    f.assert_disjoint(load_split("val"))
    f.assert_disjoint(load_split("test"))
