"""TF-IDF features, fit on the train split only, with a prompt-frequency topic-word filter.

Leakage rule (DESIGN.md, Leakage Risks): the vocabulary and IDF see only train-split rows.
`fit` asserts every row it gets is from the train split, and `assert_disjoint` checks that
the rows being transformed share no prompt with the rows the vocabulary was fit on.

Topic-word filter (BUILD_BRIEF Chunk 3; ablation 4): every prompt has ~18 answers, all on
the same topic, so words tied to one prompt's topic look like strong features but carry no
model style. A term is kept only if it appears in at least `min_prompts` distinct train
prompts. Style words ("delve", "Certainly", "—") spread across prompts and survive; topic
words ("photosynthesis") appear under one or two prompts and are dropped. min_prompts=1
turns the filter off.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def prompt_frequency(X: sp.spmatrix, prompt_ids) -> np.ndarray:
    """Number of distinct prompts each column (term) appears under."""
    codes, _ = pd.factorize(pd.Series(prompt_ids))
    P = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))))
    present = (P @ (X > 0).astype(np.float32)) > 0  # (n_prompts, n_terms)
    return np.asarray(present.sum(axis=0)).ravel()


class TfidfFeatures:
    """Word and/or char n-gram TF-IDF with the topic-word filter.

    analyzer: "word", "char" (char_wb n-grams) or "both" (side by side, each L2-normalized,
    then the whole row re-normalized).
    """

    def __init__(self, analyzer="word", word_ngram=(1, 2), char_ngram=(2, 5),
                 min_df=2, max_features=None, sublinear_tf=True, lowercase=False,
                 min_prompts=3):
        self.analyzer, self.min_prompts = analyzer, min_prompts
        common = dict(min_df=min_df, max_features=max_features, sublinear_tf=sublinear_tf,
                      lowercase=lowercase, dtype=np.float32)
        self.vecs = {}
        if analyzer in ("word", "both"):
            # Keep punctuation as tokens: em dashes, colons, bullets are style signal.
            self.vecs["word"] = TfidfVectorizer(analyzer="word", ngram_range=tuple(word_ngram),
                                                token_pattern=r"(?u)\b\w+\b|[^\w\s]", **common)
        if analyzer in ("char", "both"):
            self.vecs["char"] = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(char_ngram),
                                                **common)
        if not self.vecs:
            raise ValueError(f"analyzer must be word, char or both, got {analyzer!r}")

    def fit(self, df: pd.DataFrame) -> "TfidfFeatures":
        """df needs text, prompt_id, split. Only train-split rows are allowed."""
        splits = set(df["split"])
        assert splits == {"train"}, f"TF-IDF must be fit on the train split only, got {splits}"
        self.fit_ids_ = set(df["id"])
        self.fit_prompts_ = set(df["prompt_id"])
        self.keep_, self.stats_ = {}, {}
        for name, vec in self.vecs.items():
            X = vec.fit_transform(df["text"])
            pf = prompt_frequency(X, df["prompt_id"])
            keep = np.flatnonzero(pf >= self.min_prompts)
            self.keep_[name] = keep
            self.stats_[name] = {"vocab": X.shape[1], "kept": len(keep),
                                 "dropped_topic": X.shape[1] - len(keep)}
        return self

    def assert_disjoint(self, df: pd.DataFrame) -> None:
        """The rows being transformed must not share any row or prompt with the fit rows."""
        shared = self.fit_prompts_ & set(df["prompt_id"])
        assert not shared, f"transforming rows whose prompts the vocab was fit on: {sorted(shared)[:5]}"
        assert not (self.fit_ids_ & set(df["id"])), "transforming rows the vocab was fit on"

    def transform(self, texts) -> sp.csr_matrix:
        parts = [normalize(vec.transform(texts)[:, self.keep_[name]])
                 for name, vec in self.vecs.items()]
        X = parts[0] if len(parts) == 1 else normalize(sp.hstack(parts).tocsr())
        return X.tocsr()

    def fit_transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        return self.fit(df).transform(df["text"])

    def feature_names(self) -> np.ndarray:
        return np.concatenate([np.char.add(f"{name}:", vec.get_feature_names_out()[self.keep_[name]].astype(str))
                               for name, vec in self.vecs.items()])

    def dropped_examples(self, n=15) -> dict:
        """A few dropped (topic) terms per analyzer, to eyeball that the filter hits topics."""
        out = {}
        for name, vec in self.vecs.items():
            names = vec.get_feature_names_out()
            dropped = np.setdiff1d(np.arange(len(names)), self.keep_[name])
            idf = vec.idf_[dropped]
            out[name] = names[dropped[np.argsort(idf)[:n]]].tolist()  # most common dropped terms
        return out
