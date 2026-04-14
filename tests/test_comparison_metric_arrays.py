import numpy as np
import pandas as pd
import pytest

from meval.metrics.ComparisonMetric import ComparisonMetric
from meval.metrics.MAE import MAE
from meval.metrics.Accuracy import Accuracy


def _build_df() -> pd.DataFrame:
    idx = pd.Index([10, 11, 12, 13, 14])
    return pd.DataFrame(
        {
            "y_true": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y_pred": [0.5, 1.5, 2.5, 3.5, 4.5],
        },
        index=idx,
    )


@pytest.mark.parametrize(
    "mask",
    [
        None,
        np.array([0, 2, 4], dtype=np.int64),
        np.array([True, False, True, False, True]),
        pd.Series([True, False, True, False, True], index=[10, 11, 12, 13, 14]),
    ],
)
def test_masked_array_matches_masked_series(mask):
    df = _build_df()
    col = df["y_true"]

    arr = ComparisonMetric._masked_array(col, mask)
    series = ComparisonMetric._masked_series(col, mask)

    assert np.array_equal(arr, series.to_numpy(copy=False))


def test_masked_array_uses_pandas_alignment_for_misaligned_series_mask():
    df = _build_df()
    col = df["y_true"]

    # Reordered index should exercise the fallback path preserving pandas alignment semantics.
    mask = pd.Series([False, True, False, True, False], index=[14, 13, 12, 11, 10])

    got = ComparisonMetric._masked_array(col, mask)
    expected = col[mask].to_numpy(copy=False)

    assert np.array_equal(got, expected)


def test_get_float_y_true_array_validate_false_integer_positions():
    df = _build_df()
    mask = np.array([3, 1], dtype=np.int64)

    got = ComparisonMetric.get_float_y_true(df, mask=mask, validate=False, return_array=True)

    assert got.dtype == float
    assert np.array_equal(got, np.array([3.0, 1.0]))


def test_get_float_y_pred_array_validate_false_boolean_series_mask():
    df = _build_df()
    mask = pd.Series([True, False, True, False, True], index=df.index)

    got = ComparisonMetric.get_float_y_pred(df, mask=mask, validate=False, return_array=True)

    assert got.dtype == float
    assert np.array_equal(got, np.array([0.5, 2.5, 4.5]))


def test_bool_mask_and_index_array_are_interpreted_differently():
    df = _build_df()

    bool_mask = np.array([False, True, False, False, True], dtype=bool)
    index_mask = np.array([1, 4], dtype=np.int64)

    got_bool = ComparisonMetric.get_float_y_true(df, mask=bool_mask, validate=False, return_array=True)
    got_index = ComparisonMetric.get_float_y_true(df, mask=index_mask, validate=False, return_array=True)

    expected = np.array([1.0, 4.0])
    assert np.array_equal(got_bool, expected)
    assert np.array_equal(got_index, expected)


def test_non_boolean_pandas_series_mask_raises_type_error():
    df = _build_df()
    bad_mask = pd.Series([0, 1, 0, 1, 0], index=df.index)

    with pytest.raises(TypeError, match="boolean dtype"):
        ComparisonMetric.get_float_y_true(df, mask=bad_mask, validate=False, return_array=True)


def test_get_float_array_helpers_validate_true_match_series_helpers():
    df = _build_df()
    mask = np.array([False, True, False, True, False])

    y_true_arr = ComparisonMetric.get_float_y_true(df, mask=mask, validate=True, return_array=True)
    y_true_ser = ComparisonMetric.get_float_y_true(df, mask=mask, validate=True)
    y_pred_arr = ComparisonMetric.get_float_y_pred(df, mask=mask, validate=True, return_array=True)
    y_pred_ser = ComparisonMetric.get_float_y_pred(df, mask=mask, validate=True)

    assert isinstance(y_true_ser, pd.Series)
    assert isinstance(y_pred_ser, pd.Series)

    assert np.array_equal(y_true_arr, y_true_ser.to_numpy(dtype=float, copy=False))
    assert np.array_equal(y_pred_arr, y_pred_ser.to_numpy(dtype=float, copy=False))


def test_get_float_y_pred_array_raises_without_prediction_column():
    df = _build_df().drop(columns=["y_pred"])

    with pytest.raises(RuntimeError, match="Found no y_pred column"):
        ComparisonMetric.get_float_y_pred(df, validate=False, return_array=True)


def test_get_binary_y_pred_prob_uses_alias_column_and_mask():
    df = pd.DataFrame(
        {
            "y_true": [True, False, True, False],
            "y_prob": [0.1, 0.2, 0.8, 0.9],
        }
    )
    mask = np.array([0, 2], dtype=np.int64)

    got = ComparisonMetric.get_binary_y_pred_prob(df, mask=mask, validate=False)
    assert isinstance(got, pd.Series)

    assert np.array_equal(got.to_numpy(copy=False), np.array([0.1, 0.8]))


def test_get_binary_y_pred_prob_return_array():
    df = pd.DataFrame(
        {
            "y_true": [True, False, True, False],
            "y_prob": [0.1, 0.2, 0.8, 0.9],
        }
    )
    mask = np.array([0, 2], dtype=np.int64)

    got = ComparisonMetric.get_binary_y_pred_prob(df, mask=mask, validate=False, return_array=True)

    assert isinstance(got, np.ndarray)
    assert np.array_equal(got, np.array([0.1, 0.8]))


def test_get_binary_y_pred_return_array_from_y_pred_col():
    df = pd.DataFrame(
        {
            "y_true": [True, False, True, False],
            "y_pred": [True, True, False, False],
        }
    )
    mask = np.array([0, 3], dtype=np.int64)
    metric = Accuracy(test=True)

    got = metric.get_binary_y_pred(df, mask=mask, validate=True, return_array=True)

    assert isinstance(got, np.ndarray)
    assert got.dtype == bool
    assert np.array_equal(got, np.array([True, False]))


def test_get_binary_y_pred_return_array_from_thresholded_prob_col():
    df = pd.DataFrame(
        {
            "y_true": [True, False, True, False],
            "y_prob": [0.1, 0.7, 0.8, 0.2],
        }
    )
    mask = np.array([0, 1, 2], dtype=np.int64)
    metric = Accuracy(threshold=0.5, test=True)

    got = metric.get_binary_y_pred(df, mask=mask, validate=True, return_array=True)

    assert isinstance(got, np.ndarray)
    assert got.dtype == bool
    assert np.array_equal(got, np.array([False, True, True]))


def test_get_multiclass_y_pred_prob_with_integer_mask_preserves_order():
    df = pd.DataFrame(
        {
            "y_pred_prob_0": [0.6, 0.1, 0.2],
            "y_pred_prob_1": [0.3, 0.7, 0.5],
            "y_pred_prob_2": [0.1, 0.2, 0.3],
        }
    )
    mask = np.array([2, 0], dtype=np.int64)

    got = ComparisonMetric.get_multiclass_y_pred_prob(df, mask=mask, validate=False)
    assert isinstance(got, pd.DataFrame)

    assert list(got.columns) == ["y_pred_prob_0", "y_pred_prob_1", "y_pred_prob_2"]
    assert np.array_equal(got.to_numpy(copy=False), df.iloc[mask][got.columns].to_numpy(copy=False))


def test_get_multiclass_y_pred_prob_return_array_preserves_order():
    df = pd.DataFrame(
        {
            "y_pred_prob_0": [0.6, 0.1, 0.2],
            "y_pred_prob_1": [0.3, 0.7, 0.5],
            "y_pred_prob_2": [0.1, 0.2, 0.3],
        }
    )
    mask = np.array([2, 0], dtype=np.int64)

    got = ComparisonMetric.get_multiclass_y_pred_prob(df, mask=mask, validate=False, return_array=True)

    assert isinstance(got, np.ndarray)
    expected = df.iloc[mask][["y_pred_prob_0", "y_pred_prob_1", "y_pred_prob_2"]].to_numpy(copy=False)
    assert np.array_equal(got, expected)


def test_mae_validate_false_uses_masked_array_in_call_and_variance(monkeypatch: pytest.MonkeyPatch):
    df = _build_df()
    mask = np.array([0, 2, 4], dtype=np.int64)
    mae = MAE(test=True)

    calls = {"n": 0}
    orig = ComparisonMetric._masked_array

    def wrapped(col, m):
        calls["n"] += 1
        return orig(col, m)

    monkeypatch.setattr(ComparisonMetric, "_masked_array", staticmethod(wrapped))

    call_out = mae(df, group_mask=mask, validate=False, return_var=True)
    var_out = mae.get_variance(df, group_mask=mask, validate=False, return_val=True)

    assert isinstance(call_out, tuple)
    assert isinstance(var_out, tuple)
    val, var = call_out
    v2, var2 = var_out

    # |y_true - y_pred| is constant 0.5 for selected rows, so variance is zero.
    assert val == pytest.approx(0.5)
    assert var == pytest.approx(0.0)
    assert v2 == pytest.approx(0.5)
    assert var2 == pytest.approx(0.0)

    # Two masked array reads (y_true, y_pred) per public method call.
    assert calls["n"] >= 4
