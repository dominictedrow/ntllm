import numpy as np
import pandas as pd
import pytest

from ntllm.data.load import LABELS
from ntllm.models.blend import average_predict, features, load_scores, make_meta, standardize


def _run(tmp_path, name, scores, ids, gzip=False):
    d = tmp_path / name
    d.mkdir()
    pd.DataFrame({"id": ids}).to_csv(d / "predictions.csv", index=False)
    np.save(d / ("class_min_dist.npy" if gzip else "scores.npy"), scores)
    return d


def test_blend_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    n = 18 * 20
    y = np.array(LABELS * 20)
    ids = [f"r{i}" for i in range(n)]
    onehot = (np.array(LABELS)[None, :] == y[:, None]).astype(float)
    # "good" base: right answer on huge scale; "noise" base: random; "gzip" base: distances
    good = load_scores(_run(tmp_path, "good", 100 * onehot + rng.normal(size=(n, 18)), ids))
    noise = load_scores(_run(tmp_path, "noise", rng.normal(size=(n, 18)), ids))
    gz = load_scores(_run(tmp_path, "gz", 1 - 0.01 * onehot, ids, gzip=True))

    assert (gz.values.argmax(1) == onehot.argmax(1)).all(), "gzip distances must be negated"
    assert np.allclose(standardize(good.values).mean(1), 0)

    shuffled = ids[::-1]  # features must align by id, not by row position
    X = features({"noise": noise, "good": good}, shuffled)
    assert X.shape == (n, 36)
    assert (average_predict(X) == y[::-1]).all()      # averaging z-scores keeps the informative model on top
    assert (make_meta(1.0).fit(X, y[::-1]).predict(X) == y[::-1]).all()

    with pytest.raises(AssertionError, match="rows don't match"):
        features({"good": good}, ids[:-1])
