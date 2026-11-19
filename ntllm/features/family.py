"""Gzip family input for the encoder (DESIGN.md Features; Experiment 2).

gzip+kNN guesses the model *family* (claude / gemini / gpt / glm / kimi / qwen), and the
encoder sees it as a text prefix: "family: gpt | <passage>". A prefix needs no model
surgery and works the same at train time and in the Kaggle notebook.

Leakage: the kNN index holds only train-split rows. For the train rows themselves, a row
must not find itself or its prompt-mates (same topic, easy match), so train-row families
are leave-prompt-out: neighbors from the row's own prompt are masked. Val / test rows
just use the whole train index (their prompts are never in it).
"""
import numpy as np
import pandas as pd

from ntllm.data.load import LABELS
from ntllm.models.gzip_knn import GzipKNN

FAMILIES = ["claude", "gemini", "glm", "gpt", "kimi", "qwen"]


def family_of(label: str) -> str:
    for f in FAMILIES:
        if label.startswith(f):
            return f
    raise ValueError(f"no family for label {label!r}")


FAMILY = {label: family_of(label) for label in LABELS}


def add_prefix(texts, families) -> list[str]:
    return [f"family: {f} | {t}" for t, f in zip(texts, families)]


class GzipFamily:
    """kNN over family labels, fit on train-split rows only."""

    def __init__(self, k=7, level=9, n_jobs=None):
        self.k = k
        self.knn = GzipKNN(k=k, level=level, n_jobs=n_jobs)

    def fit(self, df: pd.DataFrame) -> "GzipFamily":
        assert set(df["split"]) == {"train"}, "gzip family index must hold train-split rows only"
        self.prompts_ = df["prompt_id"].to_numpy()
        self.fit_prompts_ = set(self.prompts_)
        self.knn.fit(df["text"].tolist(), df["model"].map(FAMILY).tolist())
        return self

    def predict_fit_rows(self) -> np.ndarray:
        """Leave-prompt-out families for the rows the index was fit on."""
        texts = [b.decode("utf-8") for b in self.knn.train_]
        D = self.knn.distances(texts)
        D[self.prompts_[:, None] == self.prompts_[None, :]] = np.inf
        return self.knn.predict_from_distances(D, self.k)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if "prompt_id" in df:  # Kaggle test has no prompt_id
            shared = self.fit_prompts_ & set(df["prompt_id"])
            assert not shared, f"rows share prompts with the gzip index: {sorted(shared)[:5]}"
        return self.knn.predict(df["text"].tolist())
