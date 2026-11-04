"""Average(col, y_class=k): the mean of col over a group's cases of class k."""

import numpy as np
import pandas as pd
import pytest

from meval import compare_groups
from meval.config import settings
from meval.metrics import Average


@pytest.fixture(autouse=True)
def _testing_config():
    settings.load_testing_config(parallel=False)


def _df(y_true) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = len(y_true)
    return pd.DataFrame({"y_true": y_true, "score": rng.normal(size=n), "site": rng.choice(["a", "b"], size=n)})


@pytest.mark.parametrize(
    ("y_true", "y_class"),
    [
        (np.random.default_rng(1).random(200) < 0.3, 0),
        (np.random.default_rng(1).random(200) < 0.3, True),
        (np.random.default_rng(2).integers(0, 3, 200), 2),
        (pd.array(np.random.default_rng(2).integers(0, 3, 200), dtype="Int64"), 1),
    ],
)
def test_class_conditional_mean_and_variance(y_true, y_class) -> None:
    df = _df(y_true)
    group = (df["site"] == "a").to_numpy()
    expected = df.loc[group & (df["y_true"] == y_class).to_numpy(), "score"]
    metric = Average("score", y_class=y_class)

    mean, var = metric.get_variance(df, group_mask=group, return_val=True)
    assert mean == pytest.approx(expected.mean())
    assert var == pytest.approx(expected.var(ddof=1) / len(expected))
    assert metric(df, group_mask=group) == pytest.approx(expected.mean())
    # Row positions (as the bootstrap passes them) select the same rows.
    assert metric(df, group_mask=np.flatnonzero(group)) == pytest.approx(expected.mean())


def test_missing_labels_belong_to_no_class() -> None:
    y_true = pd.array([0, 1, None, 1, 2, None], dtype="Int64")
    df = pd.DataFrame({"y_true": y_true, "score": [1.0, 2.0, 100.0, 4.0, 5.0, 100.0]})
    assert Average("score", y_class=1)(df) == pytest.approx(3.0)


def test_compare_groups_with_multiclass_y_class() -> None:
    df = _df(np.random.default_rng(3).integers(0, 3, 300))
    results, _ = compare_groups(df, metrics=[Average("score", y_class=k) for k in range(3)], group_by="site", min_subgroup_size=1)
    for k in range(3):
        expected = df.loc[(df["site"] == "a") & (df["y_true"] == k), "score"].mean()
        assert results.loc["site=a", f"Avg(score | y={k})"] == pytest.approx(expected)
