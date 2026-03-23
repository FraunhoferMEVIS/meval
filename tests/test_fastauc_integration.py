import importlib

import numpy as np
import pandas as pd

from meval.metrics import AUROC, MultiClassAUROC


AUROC_MODULE = importlib.import_module("meval.metrics.AUROC")
MULTI_AUROC_MODULE = importlib.import_module("meval.metrics.MultiClassAUROC")


def test_auroc_uses_fast_numba_auc(monkeypatch):
    df = pd.DataFrame(
        {
            "y_true": pd.Series([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
            "y_pred_prob": [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1],
        }
    )

    called = {"hit": False}

    def _fake_fast_numba_auc(y_true, y_score, sample_weight=None):
        called["hit"] = True
        return 0.8123

    monkeypatch.setattr(AUROC_MODULE, "fast_numba_auc", _fake_fast_numba_auc)

    result = AUROC()(df)

    assert called["hit"]
    assert result == 0.8123


def test_multiauroc_ovo_uses_fast_ovo_numba(monkeypatch):
    df = pd.DataFrame(
        {
            "y_true": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "y_pred_prob_0": [0.9, 0.8, 0.7, 0.2, 0.2, 0.3, 0.1, 0.2, 0.2],
            "y_pred_prob_1": [0.05, 0.1, 0.1, 0.7, 0.8, 0.7, 0.2, 0.2, 0.1],
            "y_pred_prob_2": [0.05, 0.1, 0.2, 0.1, 0.0, 0.0, 0.7, 0.6, 0.7],
        }
    )

    called = {"hit": False}

    def _fake_fast_ovo_auc_numba(y_true, y_score, labels=None):
        called["hit"] = True
        assert np.array_equal(np.array(labels), np.array([0, 1, 2]))
        return 0.7345

    monkeypatch.setattr(MULTI_AUROC_MODULE, "fast_ovo_auc_numba", _fake_fast_ovo_auc_numba)

    result = MultiClassAUROC(auroc_type="ovo")(df)

    assert called["hit"]
    assert result == 0.7345
