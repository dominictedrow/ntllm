"""Naive Bayes and linear SVM on TF-IDF features (DESIGN.md, Models)."""
import numpy as np
from scipy.special import softmax
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

from ntllm.data.load import LABELS


def make_model(name: str, *, alpha=0.1, C=1.0, nb_kind="complement", seed=0):
    if name == "nb":
        # ComplementNB is the usual fix for MultinomialNB's bias on text; both are offered.
        cls = ComplementNB if nb_kind == "complement" else MultinomialNB
        return cls(alpha=alpha)
    if name == "svm":
        return LinearSVC(C=C, random_state=seed, max_iter=5000)
    raise ValueError(f"model must be nb or svm, got {name!r}")


def scores(model, X) -> np.ndarray:
    """Per-class scores in LABELS order (blend features for Chunk 5).

    NB: log-probabilities. SVM: decision_function margins.
    """
    assert list(model.classes_) == LABELS, "model classes must be the 18 LABELS in order"
    if hasattr(model, "predict_log_proba"):
        return model.predict_log_proba(X)
    return model.decision_function(X)


def perplexity(model, X, y) -> float | None:
    """exp(mean negative log-likelihood of the true label). NB only: the SVM has no
    probabilities, and softmaxing margins would be a made-up number."""
    if not hasattr(model, "predict_log_proba"):
        return None
    lp = softmax(model.predict_log_proba(X), axis=1)  # renormalize (ComplementNB is loose)
    idx = np.searchsorted(LABELS, np.asarray(y))
    return float(np.exp(-np.mean(np.log(np.clip(lp[np.arange(len(idx)), idx], 1e-12, None)))))
