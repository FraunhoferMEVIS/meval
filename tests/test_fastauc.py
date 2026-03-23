import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from meval.metrics.fastauc import fast_auc, fast_numba_auc, fast_ovo_auc, fast_ovo_auc_numba


def _softmax(x):
    ex = np.exp(x - x.max(axis=1, keepdims=True))
    return ex / ex.sum(axis=1, keepdims=True)


def _assert_close_or_nan(result, expected, atol=1e-5):
    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert np.isclose(float(result), float(expected), atol=atol)


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_fastauc_binary_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=200)
    y_score = rng.random(size=200)

    expected = roc_auc_score(y_true, y_score)
    _assert_close_or_nan(fast_auc(y_true, y_score), expected)
    _assert_close_or_nan(fast_numba_auc(y_true, y_score), expected)


def test_fastauc_binary_weights_match_sklearn():
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, size=150)
    y_score = rng.random(size=150)
    weights = rng.random(size=150) + 0.1

    expected = roc_auc_score(y_true, y_score, sample_weight=weights)
    _assert_close_or_nan(fast_auc(y_true, y_score, sample_weight=weights), expected)
    _assert_close_or_nan(fast_numba_auc(y_true, y_score, sample_weight=weights), expected)


def test_fastauc_binary_degenerate_returns_nan():
    y_true = np.ones(10)
    y_score = np.linspace(0.1, 0.9, 10)

    assert np.isnan(fast_auc(y_true, y_score))
    assert np.isnan(fast_numba_auc(y_true, y_score))


@pytest.mark.parametrize("n_classes,seed", [(3, 0), (5, 1), (10, 2)])
def test_fastauc_ovo_matches_sklearn(n_classes, seed):
    rng = np.random.default_rng(seed)
    n = max(100, n_classes * 20)
    y_true = rng.integers(0, n_classes, size=n)
    y_score = _softmax(rng.standard_normal((n, n_classes)))

    expected = roc_auc_score(y_true, y_score, multi_class="ovo", average="macro")
    _assert_close_or_nan(fast_ovo_auc(y_true, y_score), expected)
    _assert_close_or_nan(fast_ovo_auc_numba(y_true, y_score), expected)


def test_fastauc_ovo_nonzero_based_labels_match_sklearn():
    rng = np.random.default_rng(11)
    labels = np.array([2, 5, 9])
    y_true = rng.choice(labels, size=90)
    y_score = _softmax(rng.standard_normal((90, 3)))

    expected = roc_auc_score(y_true, y_score, multi_class="ovo", average="macro", labels=labels)
    _assert_close_or_nan(fast_ovo_auc(y_true, y_score, labels=labels), expected)
    _assert_close_or_nan(fast_ovo_auc_numba(y_true, y_score, labels=labels), expected)


def test_fastauc_ovo_missing_label_in_ytrue_matches_sklearn():
    rng = np.random.default_rng(12)
    labels = np.array([0, 1, 2, 3])
    y_true = rng.integers(0, 3, size=80)
    y_score = _softmax(rng.standard_normal((80, 4)))

    expected = roc_auc_score(y_true, y_score, multi_class="ovo", average="macro", labels=labels)
    _assert_close_or_nan(fast_ovo_auc(y_true, y_score, labels=labels), expected)
    _assert_close_or_nan(fast_ovo_auc_numba(y_true, y_score, labels=labels), expected)


def test_fastauc_ovo_two_classes_returns_nan():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.6, 0.4],
            [0.2, 0.8],
        ]
    )

    assert np.isnan(fast_ovo_auc(y_true, y_score))
    assert np.isnan(fast_ovo_auc_numba(y_true, y_score))
