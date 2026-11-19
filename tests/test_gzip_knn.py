import numpy as np

from ntllm.models.gzip_knn import GzipKNN


def test_ncd_prefers_similar_text_and_votes():
    train = ["the cat sat on the mat " * 5, "quantum field theory renormalization " * 5,
             "the cat sat on a mat again " * 5]
    labels = ["a", "b", "a"]
    m = GzipKNN(k=1, n_jobs=1).fit(train, labels)
    D = m.distances(["the cat sat on the mat today " * 5, "renormalization of quantum fields " * 5])
    assert D.shape == (2, 3) and D.dtype == np.float32
    assert list(m.predict_from_distances(D, 1)) == ["a", "b"]
    assert list(m.predict_from_distances(D, 3)) == ["a", "a"]  # 2 of 3 neighbors are "a"
    assert m.class_min_distance(D).shape == (2, 2)


def test_parallel_matches_serial():
    train = [f"sample text number {i} " * (3 + i % 4) for i in range(12)]
    labels = [str(i % 3) for i in range(12)]
    q = [f"query {i} text " * 4 for i in range(10)]
    s = GzipKNN(n_jobs=1).fit(train, labels).distances(q)
    p = GzipKNN(n_jobs=2, chunk=3).fit(train, labels).distances(q)
    np.testing.assert_array_equal(s, p)
