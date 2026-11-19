"""Gzip + kNN classifier using normalized compression distance (NCD).

    NCD(x, y) = (C(x+y) - min(C(x), C(y))) / max(C(x), C(y)),   C = gzip-compressed length

Stdlib + numpy only: the Kaggle notebook inlines this file unchanged.
No training: fit() just stores the training texts and their compressed lengths.
"""
import gzip
import os
from multiprocessing import get_context

import numpy as np

SEP = b" "


def clen(b: bytes, level: int) -> int:
    # mtime=0 keeps the gzip header fixed so lengths are deterministic.
    return len(gzip.compress(b, compresslevel=level, mtime=0))


# Worker state, set once per process by the pool initializer.
_W = {}


def _init_worker(train_bytes, train_clen, level):
    _W["train"], _W["tlen"], _W["level"] = train_bytes, train_clen, level


def _ncd_chunk(chunk):
    train, tlen, level = _W["train"], _W["tlen"], _W["level"]
    out = np.empty((len(chunk), len(train)), dtype=np.float32)
    for i, x in enumerate(chunk):
        cx = clen(x, level)
        for j, y in enumerate(train):
            cy = tlen[j]
            out[i, j] = (clen(x + SEP + y, level) - min(cx, cy)) / max(cx, cy)
    return out


class GzipKNN:
    def __init__(self, k: int = 1, level: int = 9, n_jobs: int | None = None, chunk: int = 8):
        self.k, self.level, self.chunk = k, level, chunk
        self.n_jobs = n_jobs or os.cpu_count() or 1

    def fit(self, texts, labels):
        self.train_ = [t.encode("utf-8") for t in texts]
        self.labels_ = np.asarray(labels)
        self.classes_ = np.array(sorted(set(self.labels_)))
        self.train_clen_ = [clen(b, self.level) for b in self.train_]
        return self

    def distances(self, texts) -> np.ndarray:
        """NCD matrix, shape (len(texts), n_train). The expensive part."""
        xs = [t.encode("utf-8") for t in texts]
        chunks = [xs[i:i + self.chunk] for i in range(0, len(xs), self.chunk)]
        args = (self.train_, self.train_clen_, self.level)
        if self.n_jobs == 1:
            _init_worker(*args)
            parts = [_ncd_chunk(c) for c in chunks]
        else:
            with get_context().Pool(self.n_jobs, initializer=_init_worker, initargs=args) as pool:
                parts = pool.map(_ncd_chunk, chunks)
        return np.vstack(parts) if parts else np.empty((0, len(self.train_)), np.float32)

    def predict_from_distances(self, D: np.ndarray, k: int | None = None) -> np.ndarray:
        """Majority vote over the k nearest; ties go to the label whose neighbor came first."""
        k = k or self.k
        nearest = np.argsort(D, axis=1, kind="stable")[:, :k]
        preds = []
        for row in nearest:
            counts, best, best_n = {}, None, 0
            for j in row:
                lab = self.labels_[j]
                counts[lab] = counts.get(lab, 0) + 1
                if counts[lab] > best_n:  # strict >: earlier (closer) label wins ties
                    best, best_n = lab, counts[lab]
            preds.append(best)
        return np.array(preds)

    def class_min_distance(self, D: np.ndarray) -> np.ndarray:
        """Per-class nearest distance, shape (n, n_classes) in classes_ order. Blend feature."""
        return np.stack([D[:, self.labels_ == c].min(axis=1) for c in self.classes_], axis=1)

    def predict(self, texts) -> np.ndarray:
        return self.predict_from_distances(self.distances(texts))
