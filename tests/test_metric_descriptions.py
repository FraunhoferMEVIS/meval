"""Every metric describes itself (ComparisonMetric.description), with its threshold or class."""

import inspect

import numpy as np
import pandas as pd
import pytest

import meval.metrics as metrics_module
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
    DRMSCE,
    MAE,
    MultiClassAUROC,
    Precision,
    ProportionOfPos,
    Recall,
    Specificity,
)
from meval.metrics.ComparisonMetric import ComparisonMetric


def _binary_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    y = rng.random(60) < 0.4
    return pd.DataFrame({"y_true": y, "y_pred_prob": np.clip(0.4 * y + 0.6 * rng.random(60), 0, 1), "score": rng.random(60)})


def _multiclass_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(3), size=60)
    return pd.DataFrame({"y_true": rng.integers(0, 3, 60), **{f"y_pred_prob_{c}": probs[:, c] for c in range(3)}})


def _regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return pd.DataFrame({"y_true": rng.normal(size=60), "y_pred": rng.normal(size=60)})


METRICS = [
    (_binary_df, Accuracy(threshold=0.3)),
    (_binary_df, Recall(threshold=0.3)),
    (_binary_df, Specificity(threshold=0.3)),
    (_binary_df, Precision(threshold=0.3)),
    (_binary_df, Count()),
    (_binary_df, ProportionOfPos()),
    (_binary_df, AUROC()),
    (_binary_df, AUPRG()),
    (_binary_df, AUPRG(rec_gain_min=0.8)),
    (_binary_df, BrierScore()),
    (_binary_df, BrierScore(balanced=False)),
    (_binary_df, DRMSCE()),
    (_binary_df, CalibrationSlope()),
    (_binary_df, CalibrationIntercept()),
    (_binary_df, CalibrationIntercept(reference_score=0.1)),
    (_binary_df, Average("score")),
    (_binary_df, Average("score", y_class=1)),
    (_binary_df, BPSN_AUROC()),
    (_binary_df, BNSP_AUROC()),
    (_binary_df, ClassConditionalAUROC(0)),
    (_binary_df, ClassConditionalAUROC(1)),
    (_multiclass_df, MultiClassAUROC()),
    (_multiclass_df, MultiClassAUROC(auroc_type="ovr")),
    (_multiclass_df, Recall(per_class=True)),
    (_multiclass_df, ProportionOfPos(per_class=True)),
    (_multiclass_df, BPSN_AUROC(per_class=True)),
    (_multiclass_df, BNSP_AUROC(per_class=True)),
    (_multiclass_df, ClassConditionalAUROC(0, per_class=True)),
    (_multiclass_df, ClassConditionalAUROC(1, per_class=True)),
    (_regression_df, MAE()),
]


@pytest.mark.parametrize(("make_df", "metric"), METRICS, ids=lambda x: getattr(x, "metric_name", ""))
def test_every_resolved_metric_has_a_short_description(make_df, metric) -> None:
    for resolved in metric.resolve_metrics(make_df()):
        text = resolved.description
        assert isinstance(text, str) and "\n" not in text, resolved.metric_name
        assert 20 < len(text) <= 330, (resolved.metric_name, len(text))
        class_id = getattr(resolved, "class_id", None)
        if class_id is not None:
            assert f"class {class_id}" in text or f"class-{class_id}" in text, resolved.metric_name


def test_every_public_metric_class_overrides_description() -> None:
    classes = [obj for obj in vars(metrics_module).values()
               if inspect.isclass(obj) and issubclass(obj, ComparisonMetric) and not inspect.isabstract(obj)]
    assert len(classes) >= 18
    for cls in classes:
        assert cls.description is not ComparisonMetric.description, cls.__name__


def test_descriptions_follow_threshold_and_class() -> None:
    recall = Recall()
    assert "hard predictions" in recall.description
    recall.set_threshold(0.02)
    assert "at least 0.02" in recall.description
    false_alarms, misses = (ClassConditionalAUROC(y, per_class=True).resolve_metrics(_multiclass_df())[2] for y in (0, 1))
    assert false_alarms.metric_name == "AUROC[class=2] | y!=2"
    assert "the group's non-class-2 cases score high (false alarms)" in false_alarms.description
    assert misses.metric_name == "AUROC[class=2] | y=2"
    assert "the group's class-2 cases score low (misses)" in misses.description
