import pandas as pd
import pytest

from meval import compare_groups
from meval.config import settings
from meval.metrics import ProportionOfPos, Recall


@pytest.fixture(autouse=True)
def _testing_config():
    settings.load_testing_config(parallel=False)


def _multiclass_df() -> pd.DataFrame:
    # Argmax predictions: [0, 1, 1, 2, 0, 2]
    return pd.DataFrame(
        {
            "y_true": [0, 1, 2, 2, 0, 1],
            "y_pred_prob_0": [0.70, 0.10, 0.20, 0.15, 0.60, 0.20],
            "y_pred_prob_1": [0.20, 0.80, 0.60, 0.05, 0.30, 0.20],
            "y_pred_prob_2": [0.10, 0.10, 0.20, 0.80, 0.10, 0.60],
        }
    )


def _binary_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y_true": [True, False, True, False],
            "y_pred": [True, False, False, False],
        }
    )


def _multiclass_pred_label_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y_true": [0, 1, 2, 2, 0, 1],
            "y_pred": [0, 1, 1, 2, 0, 2],
        }
    )


def test_recall_per_class_true_binary_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="binary y_true"):
        compare_groups(df=_binary_df(), metrics=[Recall(per_class=True)], min_subgroup_size=1)


def test_recall_per_class_false_multiclass_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="only implemented for binary y_true"):
        compare_groups(df=_multiclass_df(), metrics=[Recall(per_class=False)], min_subgroup_size=1)


def test_proportion_per_class_true_binary_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="binary y_true"):
        compare_groups(df=_binary_df(), metrics=[ProportionOfPos(per_class=True)], min_subgroup_size=1)


def test_proportion_per_class_false_multiclass_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="only implemented for binary y_true"):
        compare_groups(df=_multiclass_df(), metrics=[ProportionOfPos(per_class=False)], min_subgroup_size=1)


def test_recall_per_class_multiclass_outputs_one_metric_per_class():
    all_metric_results_df, _ = compare_groups(
        df=_multiclass_df(),
        metrics=[Recall(per_class=True)],
        min_subgroup_size=1,
    )

    # Class recalls from one-vs-class with argmax predictions:
    # class 0: TP=2, FN=0 -> 1.0
    # class 1: TP=1, FN=1 -> 0.5
    # class 2: TP=1, FN=1 -> 0.5
    assert all_metric_results_df.loc["all", "Rec[class=0]"] == pytest.approx(1.0)
    assert all_metric_results_df.loc["all", "Rec[class=1]"] == pytest.approx(0.5)
    assert all_metric_results_df.loc["all", "Rec[class=2]"] == pytest.approx(0.5)


def test_recall_per_class_multiclass_supports_y_pred_without_probs():
    all_metric_results_df, _ = compare_groups(
        df=_multiclass_pred_label_df(),
        metrics=[Recall(per_class=True)],
        min_subgroup_size=1,
    )

    assert all_metric_results_df.loc["all", "Rec[class=0]"] == pytest.approx(1.0)
    assert all_metric_results_df.loc["all", "Rec[class=1]"] == pytest.approx(0.5)
    assert all_metric_results_df.loc["all", "Rec[class=2]"] == pytest.approx(0.5)


def test_proportion_per_class_multiclass_outputs_one_metric_per_class():
    all_metric_results_df, _ = compare_groups(
        df=_multiclass_df(),
        metrics=[ProportionOfPos(per_class=True)],
        min_subgroup_size=1,
    )

    # Class counts are 2/6 for each class.
    assert all_metric_results_df.loc["all", "p(y=0|G=g)"] == pytest.approx(1.0 / 3.0)
    assert all_metric_results_df.loc["all", "p(y=1|G=g)"] == pytest.approx(1.0 / 3.0)
    assert all_metric_results_df.loc["all", "p(y=2|G=g)"] == pytest.approx(1.0 / 3.0)
