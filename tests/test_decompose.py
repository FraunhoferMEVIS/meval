"""ComparisonMetric.decompose(): the metric as a weighted sum of conditional
means of per-sample scores. The defining property is checked directly: for
any group G, sum(w * mean(score[G & reference])) reproduces metric(G)."""

import numpy as np
import pandas as pd
import pytest

from meval.config import settings
from meval.metrics import (
    AUPRG,
    AUROC,
    Accuracy,
    Average,
    BNSP_AUROC,
    BPSN_AUROC,
    BrierScore,
    CalibrationIntercept,
    CalibrationSlope,
    ClassConditionalAUROC,
    Count,
    DecompositionTerm,
    DRMSCE,
    MAE,
    MultiClassAUROC,
    Precision,
    ProportionOfPos,
    Recall,
    Specificity,
)


@pytest.fixture(autouse=True)
def _testing_config():
    settings.load_testing_config(parallel=False)


N = 240


def _binary_df(with_hard_pred: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    y_true = rng.random(N) < 0.35
    prob = np.clip(0.35 * y_true + 0.65 * rng.random(N), 0.0, 1.0)
    score = rng.normal(size=N)
    score[rng.random(N) < 0.1] = np.nan  # Average() ignores missing values
    df = pd.DataFrame({"y_true": y_true, "y_pred_prob": prob, "score": score})
    if with_hard_pred:
        df["y_pred"] = (prob + 0.2 * rng.normal(size=N)) >= 0.5
    return df


def _multiclass_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    y_true = rng.choice(3, size=N, p=[0.6, 0.25, 0.15])
    logits = rng.normal(size=(N, 3)) + 1.5 * np.eye(3)[y_true]
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return pd.DataFrame(
        {"y_true": y_true, "score": rng.normal(size=N), **{f"y_pred_prob_{k}": probs[:, k] for k in range(3)}}
    )


def _regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    y_true = rng.normal(size=N)
    return pd.DataFrame({"y_true": y_true, "y_pred": y_true + rng.normal(scale=0.5, size=N)})


def _random_groups() -> list[np.ndarray]:
    rng = np.random.default_rng(3)
    groups = [np.ones(N, dtype=bool)]
    for density in (0.5, 0.2, 0.05, 0.02):  # small groups also hit empty reference sets
        for _ in range(5):
            groups.append(rng.random(N) < density)
    return [g for g in groups if g.any()]


def _from_terms(terms: list[DecompositionTerm], group: np.ndarray) -> float:
    total = 0.0
    for term in terms:
        rows = group & term.reference
        if not rows.any():
            return np.nan
        total += term.weight * float(np.mean(term.score[rows]))
    return total


DECOMPOSABLE = [
    (_binary_df, Recall()),
    (_binary_df, Specificity()),
    (_binary_df, Precision()),
    (_binary_df, Accuracy()),
    (_binary_df, ProportionOfPos()),
    (_binary_df, BrierScore()),
    (_binary_df, BrierScore(balanced=False)),
    (_binary_df, Average("score")),
    (_binary_df, Average("score", y_class=0)),
    (_binary_df, Average("score", y_class=True)),
    (_multiclass_df, Average("score", y_class=2)),
    (lambda: _binary_df(with_hard_pred=False), Recall(threshold=0.4)),
    (lambda: _binary_df(with_hard_pred=False), Specificity(threshold=0.4)),
    (_multiclass_df, Recall(per_class=True)),
    (_multiclass_df, ProportionOfPos(per_class=True)),
    (_binary_df, ClassConditionalAUROC(0)),
    (_binary_df, ClassConditionalAUROC(1)),
    (_multiclass_df, ClassConditionalAUROC(0, per_class=True)),
    (_multiclass_df, ClassConditionalAUROC(1, per_class=True)),
    (_regression_df, MAE()),
]


@pytest.mark.parametrize(("make_df", "metric"), DECOMPOSABLE, ids=lambda x: getattr(x, "metric_name", ""))
def test_decomposition_reproduces_the_metric_on_every_group(make_df, metric) -> None:
    df = make_df()
    resolved = metric.resolve_metrics(df)
    assert resolved
    for resolved_metric in resolved:
        terms = resolved_metric.decompose(df)
        assert terms is not None, resolved_metric.metric_name
        for term in terms:
            assert term.score.shape == term.reference.shape == (len(df),)
            assert term.reference.dtype == bool
            assert np.isfinite(term.score[term.reference]).all()
        for group in _random_groups():
            expected = resolved_metric(df, group_mask=group)
            np.testing.assert_allclose(_from_terms(terms, group), expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_balanced_brier_has_one_term_per_class() -> None:
    terms = BrierScore().decompose(_binary_df())
    assert terms is not None and [t.weight for t in terms] == [0.5, 0.5]
    assert not (terms[0].reference & terms[1].reference).any()


@pytest.mark.parametrize(
    ("make_df", "metric"),
    [
        (_binary_df, AUROC()),
        (_binary_df, BPSN_AUROC()),
        (_binary_df, BNSP_AUROC()),
        (_binary_df, AUPRG()),
        (_binary_df, CalibrationSlope()),
        (_binary_df, CalibrationIntercept()),
        (_binary_df, DRMSCE()),
        (_binary_df, Count()),
        (_multiclass_df, MultiClassAUROC()),
    ],
    ids=lambda x: getattr(x, "metric_name", ""),
)
def test_non_decomposable_metrics_return_none(make_df, metric) -> None:
    df = make_df()
    for resolved_metric in metric.resolve_metrics(df):
        assert resolved_metric.decompose(df) is None


def test_per_class_wrappers_are_decomposed_after_resolution() -> None:
    df = _multiclass_df()
    assert Recall(per_class=True).decompose(df) is None
    assert ProportionOfPos(per_class=True).decompose(df) is None
    assert ClassConditionalAUROC(1, per_class=True).decompose(df) is None
