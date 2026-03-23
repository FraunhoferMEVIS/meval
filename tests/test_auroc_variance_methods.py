import numpy as np
import pandas as pd
import pytest
import importlib

from meval.metrics import AUROC


AUROC_MODULE = importlib.import_module("meval.metrics.AUROC")


def _make_binary_df(y_true: list[int], y_pred_prob: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y_true": pd.Series(y_true, dtype=bool),
            "y_pred_prob": y_pred_prob,
        }
    )


def test_get_variance_newcombe_does_not_use_delong(monkeypatch):
    df = _make_binary_df(
        y_true=[1, 1, 1, 1, 0, 0, 0, 0],
        y_pred_prob=[0.9, 0.8, 0.7, 0.6, 0.55, 0.45, 0.35, 0.25],
    )

    def _delong_should_not_be_called(*args, **kwargs):
        raise AssertionError("DeLong path should not be used for method='newcombe'.")

    monkeypatch.setattr(AUROC_MODULE, "delong_roc_variance", _delong_should_not_be_called)

    auc, var = AUROC().get_variance(df, method="newcombe", return_val=True)

    assert np.isfinite(auc)
    assert np.isfinite(var)


def test_get_variance_delong_uses_delong(monkeypatch):
    df = _make_binary_df(
        y_true=[1, 1, 1, 1, 0, 0, 0, 0],
        y_pred_prob=[0.9, 0.8, 0.7, 0.6, 0.55, 0.45, 0.35, 0.25],
    )

    def _fake_delong(*args, **kwargs):
        return 0.73, 0.0123

    monkeypatch.setattr(AUROC_MODULE, "delong_roc_variance", _fake_delong)

    auc, var = AUROC().get_variance(df, method="delong", return_val=True)

    assert auc == pytest.approx(0.73)
    assert var == pytest.approx(0.0123)


def test_get_variance_auto_small_group_prefers_newcombe(monkeypatch):
    df = _make_binary_df(
        y_true=[1, 1, 1, 1, 0, 0, 0, 0],
        y_pred_prob=[0.9, 0.8, 0.7, 0.6, 0.55, 0.45, 0.35, 0.25],
    )

    def _delong_should_not_be_called(*args, **kwargs):
        raise AssertionError("AUTO should pick Newcombe for small groups (n<=50).")

    monkeypatch.setattr(AUROC_MODULE, "delong_roc_variance", _delong_should_not_be_called)

    auc, var = AUROC().get_variance(df, method="auto", return_val=True)

    assert np.isfinite(auc)
    assert np.isfinite(var)
